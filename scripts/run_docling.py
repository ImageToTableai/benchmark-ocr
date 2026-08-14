#!/usr/bin/env python3
"""Run Docling conversion over fixed sample images and emit benchmark JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.result_schema import PredictionRecord


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_converter():
    # Redirect docling caches (~/.cache/docling) onto the shared model
    # directory (see server/env.sh) so they survive restarts. Override with
    # DOCLING_HOME.
    home = os.environ.get(
        "DOCLING_HOME",
        os.path.join(os.environ.get("BENCHMARK_MODELS", "."), "home"),
    )
    os.makedirs(home, exist_ok=True)
    os.environ["HOME"] = home
    try:
        from docling.document_converter import DocumentConverter
        import docling
    except ImportError as exc:
        raise SystemExit(
            "Docling is not installed. Run: bash server/install_model_env.sh docling"
        ) from exc
    return DocumentConverter(), docling.__version__


def run_conversion(converter, image_path: Path) -> tuple[str, float]:
    start = time.perf_counter()
    result = converter.convert(str(image_path))
    text = result.document.export_to_text()
    latency_ms = (time.perf_counter() - start) * 1000
    return text or "", latency_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-samples", default=None, help="Unscored samples used only to warm runtime caches")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    converter, model_version = load_converter()
    samples_path = Path(args.samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.warmup_samples:
        for sample in read_jsonl(Path(args.warmup_samples)):
            run_conversion(converter, Path(sample["image_path"]))

    with out_path.open("w", encoding="utf-8") as handle:
        for sample in read_jsonl(samples_path):
            try:
                text, latency_ms = run_conversion(converter, Path(sample["image_path"]))
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="docling",
                    model_version=model_version,
                    text=text,
                    latency_ms=latency_ms,
                    status="ok",
                )
            except Exception as exc:
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="docling",
                    model_version=model_version,
                    status="error",
                    error=str(exc),
                )
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
