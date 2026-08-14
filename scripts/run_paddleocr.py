#!/usr/bin/env python3
"""Run PaddleOCR over a fixed sample JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.result_schema import PredictionRecord

# Persist the PaddleX model cache on the shared model directory (see
# server/env.sh) instead of ~/.paddlex (lost on restart). Override with
# PADDLE_PDX_CACHE_HOME.
os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    os.path.join(os.environ.get("BENCHMARK_MODELS", "."), "paddlex_home"),
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def import_paddleocr():
    try:
        import paddleocr
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise SystemExit(
            "PaddleOCR is required. Install with: bash server/install_model_env.sh paddleocr"
        ) from exc
    return paddleocr, PaddleOCR


def result_to_json(res) -> dict:
    if hasattr(res, "json"):
        payload = res.json
        if callable(payload):
            payload = payload()
        if isinstance(payload, dict):
            return payload.get("res") if isinstance(payload.get("res"), dict) else payload
    if isinstance(res, dict):
        return res
    return {}


def extract_text(result) -> str:
    texts: list[str] = []
    for res in result:
        payload = result_to_json(res)
        rec_texts = payload.get("rec_texts") or payload.get("texts") or []
        texts.extend(str(text).strip() for text in rec_texts if str(text).strip())
    return "\n".join(texts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-samples", default=None, help="Unscored samples used only to warm runtime caches")
    parser.add_argument("--device", default="gpu")
    args = parser.parse_args()

    paddleocr, PaddleOCR = import_paddleocr()
    samples = read_jsonl(Path(args.samples))
    out_path = Path(args.out)
    raw_dir = out_path.parent / "raw" / "paddleocr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=args.device,
    )
    if args.warmup_samples:
        for sample in read_jsonl(Path(args.warmup_samples)):
            list(ocr.predict(sample["image_path"]))

    model_version = getattr(paddleocr, "__version__", "unknown")
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            try:
                start = time.perf_counter()
                result = list(ocr.predict(sample["image_path"]))
                latency_ms = (time.perf_counter() - start) * 1000
                raw_path = raw_dir / f"{sample['sample_id']}.json"
                raw_payloads = [result_to_json(res) for res in result]
                raw_path.write_text(json.dumps(raw_payloads, ensure_ascii=False), encoding="utf-8")
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="paddleocr",
                    model_version=model_version,
                    text=extract_text(result),
                    latency_ms=latency_ms,
                    status="ok",
                    raw_output_path=raw_path.as_posix(),
                )
            except Exception as exc:
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="paddleocr",
                    model_version=model_version,
                    status="error",
                    error=str(exc),
                )
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
