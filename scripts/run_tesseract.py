#!/usr/bin/env python3
"""Run local Tesseract OCR over a fixed sample JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.result_schema import PredictionRecord


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_tesseract_cmd() -> Path | str | None:
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        return env_cmd
    return shutil.which("tesseract")


def tesseract_env() -> dict[str, str]:
    return os.environ.copy()


def get_tesseract_version(cmd: Path | str, env: dict[str, str]) -> str:
    result = subprocess.run(
        [str(cmd), "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.splitlines()[0].replace("tesseract ", "").strip()


def run_ocr(cmd: Path | str, env: dict[str, str], image_path: Path, lang: str, timeout: int) -> tuple[str, float]:
    start = time.perf_counter()
    result = subprocess.run(
        [str(cmd), image_path.as_posix(), "stdout", "-l", lang],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return result.stdout.strip(), latency_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-samples", default=None, help="Unscored samples used only to warm runtime caches")
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    tesseract_cmd = resolve_tesseract_cmd()
    if not tesseract_cmd:
        raise SystemExit(
            "tesseract binary not found. Install tesseract-ocr "
            "(e.g. `apt install tesseract-ocr`) or set TESSERACT_CMD."
        )

    samples_path = Path(args.samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = tesseract_env()
    model_version = get_tesseract_version(tesseract_cmd, env)

    rows = read_jsonl(samples_path)
    if args.warmup_samples:
        for sample in read_jsonl(Path(args.warmup_samples)):
            run_ocr(tesseract_cmd, env, Path(sample["image_path"]), args.lang, args.timeout)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in rows:
            image_path = Path(sample["image_path"])
            try:
                text, latency_ms = run_ocr(tesseract_cmd, env, image_path, args.lang, args.timeout)
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="tesseract",
                    model_version=model_version,
                    text=text,
                    latency_ms=latency_ms,
                    status="ok",
                )
            except subprocess.TimeoutExpired as exc:
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="tesseract",
                    model_version=model_version,
                    status="timeout",
                    error=str(exc),
                )
            except Exception as exc:
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="tesseract",
                    model_version=model_version,
                    status="error",
                    error=str(exc),
                )
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
