#!/usr/bin/env python3
"""Prepare fixed SROIE sample JSONL files from Hugging Face.

Raw images are saved under ignored local data directories. The resulting sample
JSONL can be tracked when it represents a stable benchmark subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from itertools import islice
from typing import Any


REPO_ID = "jsdnrs/ICDAR2019-SROIE"
DATASET_ID = "sroie_2019"
DATASET_VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def import_load_dataset():
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "Python package 'datasets' is required. Install the current model env first: "
            "bash server/install_model_env.sh <model>"
        ) from exc
    return load_dataset


def import_pyarrow():
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Python package 'pyarrow' is required for --local-parquet. Install project "
            "requirements through: bash server/install_model_env.sh <model>"
        ) from exc
    return pq


def import_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise SystemExit(
            "Python package 'huggingface_hub' is required for --download-parquet. "
            "Install project requirements through: bash server/install_model_env.sh <model>"
        ) from exc
    return hf_hub_download


def import_requests():
    try:
        import requests
    except Exception as exc:
        raise SystemExit(
            "Python package 'requests' is required for direct parquet download. "
            "Install project requirements through: bash server/install_model_env.sh <model>"
        ) from exc
    return requests


def stable_sample_id(split: str, key: str, index: int) -> str:
    if key:
        safe_key = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in key)
        return f"sroie_{split}_{safe_key}"
    digest = hashlib.sha1(f"{split}:{index}".encode("utf-8")).hexdigest()[:10]
    return f"sroie_{split}_{digest}"


def full_text_from_words(words: list[Any]) -> str:
    return "\n".join(str(word).strip() for word in words if str(word).strip())


def save_image(row: dict[str, Any], image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = row["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(image_path)


def configure_hf_env(endpoint: str | None, timeout: int) -> None:
    os.environ.setdefault("HF_HOME", str(Path("datasets/downloads/hf_home").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(timeout))
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")


def download_parquet_with_hf_cli(split: str, out_dir: Path) -> Path:
    filename = f"data/{split}-00000-of-00001.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download = import_hf_hub_download()
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=out_dir,
        )
    )


def download_parquet_direct(split: str, out_dir: Path, endpoint: str | None, timeout: int) -> Path:
    requests = import_requests()
    filename = f"data/{split}-00000-of-00001.parquet"
    endpoint = (endpoint or "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/datasets/{REPO_ID}/resolve/main/{filename}"
    out_path = out_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    headers = {}
    existing = tmp_path.stat().st_size if tmp_path.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"

    with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
        if response.status_code == 416 and tmp_path.exists():
            tmp_path.rename(out_path)
            return out_path
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        downloaded = existing if mode == "ab" else 0
        with tmp_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded and downloaded % (25 * 1024 * 1024) < 1024 * 1024:
                        print(f"downloaded {downloaded / 1024 / 1024:.1f} MB")
    tmp_path.rename(out_path)
    return out_path


def rows_from_dataset_viewer(split: str, offset: int, limit: int, timeout: int):
    requests = import_requests()
    fetched = 0
    current_offset = offset
    while fetched < limit:
        batch_len = min(100, limit - fetched)
        response = requests.get(
            DATASET_VIEWER_ROWS_URL,
            params={
                "dataset": REPO_ID,
                "config": "default",
                "split": split,
                "offset": current_offset,
                "length": batch_len,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows") or []
        if not rows:
            break
        for item in rows:
            yield item["row"]
            fetched += 1
        current_offset += len(rows)


def download_viewer_image(row: dict[str, Any], image_path: Path, timeout: int) -> None:
    requests = import_requests()
    image = row.get("image")
    if not isinstance(image, dict) or not image.get("src"):
        raise ValueError("Dataset viewer row does not include image.src")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(image["src"], timeout=timeout)
    response.raise_for_status()
    image_path.write_bytes(response.content)


def rows_from_local_parquet(path: Path, limit: int):
    pq = import_pyarrow()
    table = pq.read_table(path)
    for row in table.slice(0, limit).to_pylist():
        yield row


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize rows from datasets.load_dataset or local parquet."""
    image = row.get("image")
    if isinstance(image, dict) and "bytes" in image:
        from io import BytesIO
        from PIL import Image

        row = dict(row)
        row["image"] = Image.open(BytesIO(image["bytes"]))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", help="Hugging Face split, usually train or test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--samples-out", default="datasets/samples/sroie_test_50.jsonl")
    parser.add_argument("--image-dir", default="datasets/processed/sroie_2019/images")
    parser.add_argument("--cache-dir", default="datasets/downloads/hf_cache")
    parser.add_argument("--streaming", action="store_true", help="Stream rows instead of preparing the full split")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--hf-timeout", type=int, default=900)
    parser.add_argument("--local-parquet", default=None, help="Use an already downloaded parquet file")
    parser.add_argument("--download-parquet", action="store_true", help="Download only the split parquet with hf CLI first")
    parser.add_argument("--direct-download", action="store_true", help="Directly stream the split parquet from endpoint/resolve")
    parser.add_argument("--viewer-rows", action="store_true", help="Use Hugging Face Dataset Viewer rows API")
    parser.add_argument("--viewer-offset", type=int, default=0)
    parser.add_argument("--parquet-dir", default="datasets/downloads/sroie_2019")
    args = parser.parse_args()

    configure_hf_env(args.hf_endpoint, args.hf_timeout)

    if args.local_parquet:
        rows = rows_from_local_parquet(Path(args.local_parquet), args.limit)
    elif args.viewer_rows:
        rows = rows_from_dataset_viewer(args.split, args.viewer_offset, args.limit, args.hf_timeout)
    elif args.direct_download:
        parquet_path = download_parquet_direct(args.split, Path(args.parquet_dir), args.hf_endpoint, args.hf_timeout)
        rows = rows_from_local_parquet(parquet_path, args.limit)
    elif args.download_parquet:
        parquet_path = download_parquet_with_hf_cli(args.split, Path(args.parquet_dir))
        rows = rows_from_local_parquet(parquet_path, args.limit)
    else:
        load_dataset = import_load_dataset()
        dataset = load_dataset(REPO_ID, split=args.split, cache_dir=args.cache_dir, streaming=args.streaming)
        rows = islice(dataset, args.limit) if args.streaming else dataset.select(range(min(args.limit, len(dataset))))

    samples_out = Path(args.samples_out)
    image_dir = Path(args.image_dir)
    samples_out.parent.mkdir(parents=True, exist_ok=True)

    with samples_out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row = normalize_row(row)
            key = str(row.get("key") or "")
            sample_id = stable_sample_id(args.split, key, index)
            image_path = image_dir / f"{sample_id}.png"
            if args.viewer_rows:
                download_viewer_image(row, image_path, args.hf_timeout)
            else:
                save_image(row, image_path)
            sample = {
                "sample_id": sample_id,
                "dataset_id": DATASET_ID,
                "source_repo": REPO_ID,
                "source_split": args.split,
                "source_key": key,
                "image_path": image_path.as_posix(),
                "gt_text": full_text_from_words(row.get("words") or []),
                "gt_fields": row.get("entities") or {},
                "doc_type": "receipt",
                "language": "en",
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(samples_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
