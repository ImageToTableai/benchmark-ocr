#!/usr/bin/env python3
"""Run docTR (Document Text Recognition) over a fixed sample JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.result_schema import PredictionRecord

# Persist docTR model weights on the shared model directory (see server/env.sh)
# instead of ~/.cache/doctr (lost on restart). Override with DOCTR_CACHE_DIR.
os.environ.setdefault(
    "DOCTR_CACHE_DIR",
    os.path.join(os.environ.get("BENCHMARK_MODELS", "."), "doctr_cache"),
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_predictor(gpu: bool):
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except ImportError:
        raise SystemExit(
            "python-doctr not installed. Run: bash server/install_model_env.sh doctr"
        )
    predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
    if gpu:
        try:
            import torch

            if torch.cuda.is_available():
                return predictor.to("cuda")
        except ImportError:
            pass
    return predictor


def run_ocr(predictor, image_path: Path, timeout: int) -> tuple[str, float]:
    from doctr.io import DocumentFile

    start = time.perf_counter()
    doc = DocumentFile.from_images(str(image_path))
    result = predictor(doc)
    latency_ms = (time.perf_counter() - start) * 1000

    lines = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.value:
                        lines.append(word.value)
            text = "\n".join(lines) if lines else page.synthesize() if hasattr(page, "synthesize") else " ".join(lines)
    return text, latency_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-samples", default=None, help="Unscored samples used only to warm runtime caches")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    predictor = load_predictor(gpu=True)
    import doctr
    model_version = doctr.__version__

    samples_path = Path(args.samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(samples_path)
    if args.warmup_samples:
        for sample in read_jsonl(Path(args.warmup_samples)):
            run_ocr(predictor, Path(sample["image_path"]), args.timeout)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in rows:
            image_path = Path(sample["image_path"])
            try:
                text, latency_ms = run_ocr(predictor, image_path, args.timeout)
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="doctr",
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
                    model="doctr",
                    model_version=model_version,
                    status="error",
                    error=str(exc),
                )
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
