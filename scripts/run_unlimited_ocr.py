#!/usr/bin/env python3
"""Run Unlimited-OCR through an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.result_schema import PredictionRecord


DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
REF_OPEN_RE = re.compile(r"<\|ref\|>", re.DOTALL)
REF_CLOSE_RE = re.compile(r"<\|/ref\|>", re.DOTALL)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def import_openai():
    try:
        from openai import OpenAI
    except Exception as exc:
        raise SystemExit(
            "OpenAI client is required. Install with: bash server/install_model_env.sh unlimited_ocr"
        ) from exc
    return OpenAI


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.as_posix())[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def clean_unlimited_text(value: str) -> str:
    value = DET_RE.sub("", value)
    value = REF_OPEN_RE.sub("", value)
    value = REF_CLOSE_RE.sub("", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-samples", default=None, help="Unscored samples used only to warm runtime caches")
    parser.add_argument("--base-url", default=os.environ.get("UNLIMITED_OCR_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR"))
    parser.add_argument("--api-key", default=os.environ.get("UNLIMITED_OCR_API_KEY", "EMPTY"))
    parser.add_argument("--prompt", default="<image>document parsing.")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    OpenAI = import_openai()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    samples = read_jsonl(Path(args.samples))
    out_path = Path(args.out)
    raw_dir = out_path.parent / "raw" / "unlimited_ocr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.warmup_samples:
        for sample in read_jsonl(Path(args.warmup_samples)):
            client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": args.prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url(Path(sample["image_path"]))}},
                        ],
                    }
                ],
                max_tokens=args.max_tokens,
                temperature=0.0,
                extra_body={"skip_special_tokens": False, "vllm_xargs": {"ngram_size": 35, "window_size": 128}},
            )

    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            try:
                start = time.perf_counter()
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": args.prompt},
                                {"type": "image_url", "image_url": {"url": image_data_url(Path(sample["image_path"]))}},
                            ],
                        }
                    ],
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    extra_body={
                        "skip_special_tokens": False,
                        "vllm_xargs": {"ngram_size": 35, "window_size": 128},
                    },
                )
                latency_ms = (time.perf_counter() - start) * 1000
                text = response.choices[0].message.content or ""
                raw_path = raw_dir / f"{sample['sample_id']}.json"
                raw_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="unlimited_ocr",
                    model_version=args.model,
                    text=clean_unlimited_text(text),
                    latency_ms=latency_ms,
                    status="ok",
                    raw_output_path=raw_path.as_posix(),
                )
            except Exception as exc:
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="unlimited_ocr",
                    model_version=args.model,
                    status="error",
                    error=str(exc),
                )
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
