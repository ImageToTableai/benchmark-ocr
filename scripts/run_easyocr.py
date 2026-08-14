#!/usr/bin/env python3
"""Run EasyOCR over a fixed sample JSONL file."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
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


def load_reader(lang_list: list[str], gpu: bool):
    try:
        import easyocr
    except ImportError:
        raise SystemExit(
            "easyocr not installed. Run: bash server/install_model_env.sh easyocr"
        )
    # Persist EasyOCR weights on the shared model directory (see server/env.sh)
    # instead of ~/.EasyOCR (lost on restart). Override with EASYOCR_MODEL_DIR.
    model_dir = os.environ.get(
        "EASYOCR_MODEL_DIR",
        os.path.join(os.environ.get("BENCHMARK_MODELS", "."), "easyocr"),
    )
    os.makedirs(model_dir, exist_ok=True)
    return easyocr.Reader(lang_list, gpu=gpu, model_storage_directory=model_dir)


class OcrTimeoutError(TimeoutError):
    pass


def _raise_timeout(signum, frame):
    raise OcrTimeoutError("EasyOCR page timed out")


def run_ocr(reader, image_path: Path, timeout: int) -> tuple[str, float]:
    start = time.perf_counter()
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout)
    try:
        results = reader.readtext(str(image_path), detail=0)
        latency_ms = (time.perf_counter() - start) * 1000
        text = "\n".join(results) if results else ""
        return text, latency_ms
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_ocr_process_worker(conn, image_path: str, lang_list: list[str], gpu: bool) -> None:
    try:
        reader = load_reader(lang_list, gpu)
        results = reader.readtext(image_path, detail=0)
        text = "\n".join(results) if results else ""
        conn.send({"ok": True, "text": text})
    except Exception as exc:
        conn.send({"ok": False, "error": str(exc)})
    finally:
        conn.close()


def run_ocr_in_process(image_path: Path, lang_list: list[str], gpu: bool, timeout: int) -> tuple[str, float]:
    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_run_ocr_process_worker,
        args=(child_conn, str(image_path), lang_list, gpu),
    )
    proc.start()
    child_conn.close()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise OcrTimeoutError(f"EasyOCR page timed out after {timeout}s")
    latency_ms = (time.perf_counter() - start) * 1000
    if not parent_conn.poll():
        raise RuntimeError(f"EasyOCR worker exited without a result, exitcode={proc.exitcode}")
    payload = parent_conn.recv()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "EasyOCR worker failed"))
    return str(payload.get("text", "")), latency_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup-samples", default=None, help="Unscored samples used only to warm runtime caches")
    parser.add_argument("--lang", default="en", help="Language code (en, ch_sim, etc.)")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--timeout-mode",
        choices=("signal", "process"),
        default="signal",
        help="Use process mode only for smoke/debug runs where EasyOCR may hang inside native code.",
    )
    args = parser.parse_args()

    lang_list = [lang.strip() for lang in args.lang.split(",")]
    reader = None if args.timeout_mode == "process" else load_reader(lang_list, args.gpu)
    model_version = "1.7.2"

    samples_path = Path(args.samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(samples_path)
    if args.warmup_samples:
        for sample in read_jsonl(Path(args.warmup_samples)):
            try:
                if args.timeout_mode == "process":
                    run_ocr_in_process(Path(sample["image_path"]), lang_list, args.gpu, args.timeout)
                else:
                    run_ocr(reader, Path(sample["image_path"]), args.timeout)
            except Exception:
                # Warmup samples are unscored; a hang/timeout here must not kill the run.
                pass
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in rows:
            image_path = Path(sample["image_path"])
            try:
                if args.timeout_mode == "process":
                    text, latency_ms = run_ocr_in_process(image_path, lang_list, args.gpu, args.timeout)
                else:
                    text, latency_ms = run_ocr(reader, image_path, args.timeout)
                record = PredictionRecord(
                    run_id=args.run_id,
                    sample_id=sample["sample_id"],
                    dataset_id=sample["dataset_id"],
                    model="easyocr",
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
                    model="easyocr",
                    model_version=model_version,
                    status="error",
                    error=str(exc),
                )
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
