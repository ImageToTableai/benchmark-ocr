#!/usr/bin/env python3
"""Prepare fixed sample JSONL files from Hugging Face datasets.

This script is intentionally tolerant of dataset schema differences. Each
dataset keeps its own raw format, but runners only consume the normalized sample
JSONL fields.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any


TEXT_KEYS = (
    "gt_text",
    "ground_truth_text",
    "full_text",
    "text",
    "transcription",
    "transcript",
    "ocr",
    "words",
    "tokens",
)

FIELD_KEYS = (
    "gt_fields",
    "fields",
    "entities",
    "gt_parse",
    "parse",
    "annotations",
)

IMAGE_KEYS = (
    "image",
    "page_image",
    "document",
    "img",
    "image_path",
    "file_name",
    "filename",
    "path",
)


def import_load_dataset():
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "Python package 'datasets' is required. Install a dataset env first: "
            "bash server/install_model_env.sh datasets"
        ) from exc
    return load_dataset


def import_requests():
    try:
        import requests
    except Exception as exc:
        raise SystemExit(
            "Python package 'requests' is required. Install a dataset env first: "
            "bash server/install_model_env.sh datasets"
        ) from exc
    return requests


def import_pil_image():
    try:
        from PIL import Image
    except Exception as exc:
        raise SystemExit(
            "Python package 'Pillow' is required. Install a dataset env first: "
            "bash server/install_model_env.sh datasets"
        ) from exc
    return Image


def configure_hf_env(endpoint: str | None, timeout: int) -> None:
    os.environ.setdefault("HF_HOME", str(Path("datasets/downloads/hf_home").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(timeout))
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")


def stable_sample_id(dataset_id: str, split: str, row: dict[str, Any], index: int) -> str:
    for key in ("id", "uid", "key", "file_name", "filename", "image_path", "path"):
        value = row.get(key)
        if value:
            raw = str(value)
            stem = Path(raw).stem if "/" in raw or "." in raw else raw
            safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)
            return f"{dataset_id}_{split}_{safe}"
    digest = hashlib.sha1(json.dumps(row, default=str, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{dataset_id}_{split}_{index:06d}_{digest}"


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value or value[0] not in "{[":
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def flatten_text(value: Any) -> str:
    value = parse_jsonish(value)
    parts: list[str] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
        elif isinstance(item, (int, float)):
            parts.append(str(item))
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)
    return "\n".join(parts)


def extract_ground_truth(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for key in ("ground_truth", "label", "target"):
        if key in row:
            parsed = parse_jsonish(row[key])
            if isinstance(parsed, dict):
                fields = parsed.get("gt_parse") if isinstance(parsed.get("gt_parse"), dict) else parsed
                return flatten_text(parsed), fields
            text = flatten_text(parsed)
            if text:
                return text, {}

    # mychen76/invoices-and-receipts_ocr_v1: parsed_data.json is nested JSON (sometimes Python dict syntax)
    if "parsed_data" in row:
        parsed = parse_jsonish(row["parsed_data"])
        if isinstance(parsed, dict):
            inner_raw = parsed.get("json", "")
            inner = parse_jsonish(inner_raw)
            if not isinstance(inner, dict) and isinstance(inner_raw, str):
                try:
                    inner = ast.literal_eval(inner_raw)
                except (ValueError, SyntaxError):
                    pass
            if isinstance(inner, dict):
                if "header" in inner and ("items" in inner or "summary" in inner):
                    header = inner.get("header", {})
                elif any(k not in ("items", "summary", "header") for k in inner):
                    header = inner
                else:
                    header = inner.get("header", {})
                return flatten_text(inner), header

    # mychen76: raw_data.ocr_words for text
    if "raw_data" in row:
        raw = parse_jsonish(row["raw_data"])
        if isinstance(raw, dict):
            ocr_text = raw.get("ocr_words", "")
            parsed_text = parse_jsonish(ocr_text)
            if isinstance(parsed_text, list):
                return "\n".join(parsed_text), {}

    for key in FIELD_KEYS:
        if key in row:
            parsed = parse_jsonish(row[key])
            if isinstance(parsed, dict):
                return flatten_text(parsed), parsed

    for key in TEXT_KEYS:
        if key in row:
            text = flatten_text(row[key])
            if text:
                return text, {}

    return "", {}


def image_from_dict(value: dict[str, Any], timeout: int):
    Image = import_pil_image()
    if value.get("bytes"):
        return Image.open(BytesIO(value["bytes"]))
    if value.get("path"):
        return Image.open(value["path"])
    if value.get("src"):
        requests = import_requests()
        response = requests.get(value["src"], timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    return None


def save_image_from_row(row: dict[str, Any], image_path: Path, timeout: int) -> bool:
    Image = import_pil_image()
    image_path.parent.mkdir(parents=True, exist_ok=True)

    for key in IMAGE_KEYS:
        if key not in row or row[key] in (None, ""):
            continue
        value = row[key]

        if hasattr(value, "save"):
            image = value
            if getattr(image, "mode", None) != "RGB":
                image = image.convert("RGB")
            image.save(image_path)
            return True

        if isinstance(value, dict):
            image = image_from_dict(value, timeout)
            if image is not None:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path)
                return True

        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                requests = import_requests()
                response = requests.get(value, timeout=timeout)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path)
                return True
            source_path = Path(value)
            if source_path.exists():
                suffix = source_path.suffix.lower()
                target_path = image_path.with_suffix(suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".png")
                shutil.copyfile(source_path, target_path)
                if target_path != image_path:
                    image = Image.open(target_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(image_path)
                return True

    return False


def load_rows(repo: str, name: str | None, split: str, streaming: bool, cache_dir: str | None):
    load_dataset = import_load_dataset()
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if name:
        return load_dataset(repo, name, **kwargs)
    return load_dataset(repo, **kwargs)


def iter_limited(rows: Iterable[dict[str, Any]], limit: int):
    return islice(rows, limit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--name", default=None, help="Optional Hugging Face config/subset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--doc-type", default="mixed")
    parser.add_argument("--language", default="unknown")
    parser.add_argument("--samples-out", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--cache-dir", default="datasets/downloads/hf_cache")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--hf-timeout", type=int, default=900)
    args = parser.parse_args()

    configure_hf_env(args.hf_endpoint, args.hf_timeout)
    rows = load_rows(args.repo, args.name, args.split, args.streaming, args.cache_dir)

    samples_out = Path(args.samples_out)
    image_dir = Path(args.image_dir)
    samples_out.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with samples_out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(iter_limited(rows, args.limit * 5)):
            row = dict(row)
            sample_id = stable_sample_id(args.dataset_id, args.split, row, index)
            image_path = image_dir / f"{sample_id}.png"
            try:
                has_image = save_image_from_row(row, image_path, args.hf_timeout)
            except Exception as exc:
                skipped += 1
                print(f"skip {sample_id}: image error: {exc}")
                continue
            if not has_image:
                skipped += 1
                print(f"skip {sample_id}: no image column found")
                continue

            gt_text, gt_fields = extract_ground_truth(row)
            sample = {
                "sample_id": sample_id,
                "dataset_id": args.dataset_id,
                "source_repo": args.repo,
                "source_config": args.name,
                "source_split": args.split,
                "image_path": image_path.as_posix(),
                "gt_text": gt_text,
                "gt_fields": gt_fields,
                "doc_type": args.doc_type,
                "language": args.language,
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.limit:
                break

    if written == 0:
        raise SystemExit(f"No samples written for {args.dataset_id}; skipped={skipped}")

    print(samples_out)
    print(f"written={written} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
