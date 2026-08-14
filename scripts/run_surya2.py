#!/usr/bin/env python3
"""Run the current Surya 2 VLM OCR API over fixed sample images."""

from __future__ import annotations

import argparse
import html
import json
import re
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


def html_to_text(value: str) -> str:
    """Flatten Surya's block HTML into the benchmark's plain-text field."""
    text = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(text).split())


def load_surya_pipeline():
    try:
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
    except ImportError as exc:
        raise SystemExit(
            "Current Surya API is unavailable. Install surya-ocr==0.22.1. "
            f"Import error: {exc}"
        ) from exc

    # The manager starts or attaches to the vLLM/llama.cpp backend on first use.
    manager = SuryaInferenceManager(lazy=True)
    try:
        manager.start()
    except Exception as exc:
        manager.stop()
        raise SystemExit(
            "Surya backend is unavailable. Provide Docker + NVIDIA runtime, "
            "llama-server, or SURYA_INFERENCE_URL. "
            f"Backend error: {exc}"
        ) from exc
    return manager, RecognitionPredictor(manager)


def run_ocr_surya(predictor, image_path: Path) -> tuple[str, float]:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    start = time.perf_counter()
    predictions = predictor([image], full_page=True)
    latency_ms = (time.perf_counter() - start) * 1000

    lines = []
    for block in predictions[0].blocks:
        if block.skipped or block.error:
            continue
        text = html_to_text(block.html)
        if text:
            lines.append(text)
    return "\n".join(lines), latency_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-samples", default=None, help="Unscored samples used only to warm runtime caches")
    parser.add_argument("--lang", default="en", help="Retained for runner compatibility")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    manager, predictor = load_surya_pipeline()
    model_version = "0.22.1"
    samples_path = Path(args.samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        rows = read_jsonl(samples_path)
        if args.warmup_samples:
            for sample in read_jsonl(Path(args.warmup_samples)):
                run_ocr_surya(predictor, Path(sample["image_path"]))
        with out_path.open("w", encoding="utf-8") as handle:
            for sample in rows:
                image_path = Path(sample["image_path"])
                try:
                    text, latency_ms = run_ocr_surya(predictor, image_path)
                    record = PredictionRecord(
                        run_id=args.run_id,
                        sample_id=sample["sample_id"],
                        dataset_id=sample["dataset_id"],
                        model="surya2",
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
                        model="surya2",
                        model_version=model_version,
                        status="error",
                        error=str(exc),
                    )
                handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        manager.stop()

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
