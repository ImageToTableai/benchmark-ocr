#!/usr/bin/env python3
"""Run PaddleOCR-VL over a fixed sample JSONL file."""

from __future__ import annotations

import argparse
import json
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


def import_paddleocr_vl():
    try:
        import paddleocr
        from paddleocr import PaddleOCRVL
    except Exception as exc:
        raise SystemExit(
            "PaddleOCR-VL is required. Install with: bash server/install_model_env.sh paddleocr_vl"
        ) from exc
    return paddleocr, PaddleOCRVL


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
    blocks: list[str] = []
    for res in result:
        payload = result_to_json(res)
        for block in payload.get("parsing_res_list") or []:
            content = str(block.get("block_content") or "").strip()
            if content:
                blocks.append(content)
    return "\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-samples", default=None, help="Unscored samples used only to warm runtime caches")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--vl-rec-backend", default=None)
    parser.add_argument("--vl-rec-server-url", default=None)
    parser.add_argument("--vl-rec-api-model-name", default=None)
    parser.add_argument("--vl-rec-api-key", default=None)
    args = parser.parse_args()

    paddleocr, PaddleOCRVL = import_paddleocr_vl()
    samples = read_jsonl(Path(args.samples))
    out_path = Path(args.out)
    raw_dir = out_path.parent / "raw" / "paddleocr_vl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs = {"device": args.device}
    if args.vl_rec_backend:
        pipeline_kwargs["vl_rec_backend"] = args.vl_rec_backend
    if args.vl_rec_server_url:
        pipeline_kwargs["vl_rec_server_url"] = args.vl_rec_server_url
    if args.vl_rec_api_model_name:
        pipeline_kwargs["vl_rec_api_model_name"] = args.vl_rec_api_model_name
    if args.vl_rec_api_key:
        pipeline_kwargs["vl_rec_api_key"] = args.vl_rec_api_key
    pipeline = PaddleOCRVL(**pipeline_kwargs)
    model_version = getattr(paddleocr, "__version__", "unknown")
    if args.warmup_samples:
        for sample in read_jsonl(Path(args.warmup_samples)):
            list(pipeline.predict(sample["image_path"]))

    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            try:
                start = time.perf_counter()
                result = list(pipeline.predict(sample["image_path"]))
                latency_ms = (time.perf_counter() - start) * 1000
                raw_path = raw_dir / f"{sample['sample_id']}.json"
                raw_payloads = [result_to_json(res) for res in result]
                raw_path.write_text(json.dumps(raw_payloads, ensure_ascii=False), encoding="utf-8")
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="paddleocr_vl",
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
                    model="paddleocr_vl",
                    model_version=model_version,
                    status="error",
                    error=str(exc),
                )
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
