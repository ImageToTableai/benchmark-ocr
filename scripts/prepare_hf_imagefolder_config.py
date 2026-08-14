#!/usr/bin/env python3
"""Download a fixed Hugging Face imagefolder configuration with OCR text GT.

This avoids the optional ``datasets`` package at collection time. It is meant
for repositories that keep one ``metadata.jsonl`` beside images in each
configuration directory, such as ``nrl-ai/vn-ocr-documents-eval``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def resolve_url(repo: str, revision: str, relative_path: str) -> str:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/"))
    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in relative_path.split("/"))
    return f"https://huggingface.co/datasets/{quoted_repo}/resolve/{urllib.parse.quote(revision, safe='')}/{quoted_path}"


def fetch_bytes(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "benchmark-ocr-dataset-audit/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_metadata(repo: str, revision: str, config: str, timeout: int) -> list[dict[str, Any]]:
    url = resolve_url(repo, revision, f"{config}/metadata.jsonl")
    raw = fetch_bytes(url, timeout).decode("utf-8")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No metadata records found at {url}")
    return records


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def download_config(
    *,
    dataset_id: str,
    repo: str,
    revision: str,
    config: str,
    limit: int,
    samples_out: Path,
    image_dir: Path,
    doc_type: str,
    language: str,
    timeout: int,
) -> int:
    records = load_metadata(repo, revision, config, timeout)
    selected = records[:limit] if limit else records
    if not selected:
        raise ValueError("The requested limit selected no records")

    samples_out.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    temporary_samples_out = samples_out.with_name(f".{samples_out.name}.tmp")
    written = 0

    try:
        with temporary_samples_out.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(selected):
                filename = str(record.get("file_name", "")).strip()
                text = str(record.get("text", "")).strip()
                if not filename or not text:
                    raise ValueError(f"Record {index} lacks file_name or text")

                doc_id = str(record.get("doc_id") or Path(filename).stem)
                sample_id = f"{safe_id(dataset_id)}_{safe_id(config)}_{safe_id(doc_id)}"
                image_path = image_dir / f"{sample_id}{Path(filename).suffix.lower() or '.png'}"
                image_path.write_bytes(fetch_bytes(resolve_url(repo, revision, f"{config}/{filename}"), timeout))

                sample = {
                    "sample_id": sample_id,
                    "dataset_id": dataset_id,
                    "source_repo": repo,
                    "source_config": config,
                    "source_split": "test",
                    "source_revision": revision,
                    "image_path": image_path.as_posix(),
                    "gt_text": text,
                    "gt_fields": {},
                    "doc_type": doc_type,
                    "language": language,
                    "source_document_id": doc_id,
                    "source_record_license": record.get("license"),
                    "source_generation_method": record.get("gen_method"),
                }
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
        temporary_samples_out.replace(samples_out)
    except Exception:
        temporary_samples_out.unlink(missing_ok=True)
        raise
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit", type=int, default=0, help="0 means every metadata record")
    parser.add_argument("--samples-out", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--doc-type", default="mixed")
    parser.add_argument("--language", default="unknown")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    written = download_config(
        dataset_id=args.dataset_id,
        repo=args.repo,
        revision=args.revision,
        config=args.config,
        limit=args.limit,
        samples_out=Path(args.samples_out),
        image_dir=Path(args.image_dir),
        doc_type=args.doc_type,
        language=args.language,
        timeout=args.timeout,
    )
    print(args.samples_out)
    print(f"written={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
