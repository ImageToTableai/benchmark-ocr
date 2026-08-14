#!/usr/bin/env python3
"""Concurrent LLM field postprocessor: extract receipt fields from OCR text via an OpenAI-compatible LLM API.

Concurrency model:
  - ThreadPoolExecutor with configurable workers (default 30)
  - exponential-backoff retry, max N attempts (wait = min(60, attempt*10))
  - resume-aware: sample_ids already present in the output file are skipped
  - per-sample fallback on error (never crashes the whole run)
  - incremental atomic flush for progress visibility + crash safety

The API key is read from the LLM_API_KEY environment variable. Set
LLM_BASE_URL to any OpenAI-compatible endpoint (e.g. https://api.openai.com/v1)
and LLM_MODEL to the model name.

Usage:
  export LLM_BASE_URL=https://api.openai.com/v1
  export LLM_MODEL=gpt-4o-mini
  python3 scripts/run_llm_field_postprocessor.py \
    --predictions results/<run_id>/<model>_predictions.jsonl \
    --out results/<run_id>/<model>_llm_<llm>_fields.jsonl \
    --dataset sroie --workers 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_WORKERS = 30
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
FLUSH_EVERY = 20

SYSTEM_PROMPTS = {
    "sroie": (
        "You extract receipt fields from OCR text. Return ONLY a JSON object "
        "with exactly these keys: company, date, address, total. "
        "Preserve the original capitalization and spelling exactly as written. "
        "Use an empty string for any field you cannot find. "
        "Do not add any other text, keys, or markdown."
    ),
    "cord": (
        "You extract receipt fields from OCR text. Return ONLY a JSON object "
        "with exactly these keys: "
        "menu (object with keys nm, num, cnt, price, itemsubtotal), "
        "sub_total (object with keys subtotal_price, discount_price, tax_price), "
        "total (object with keys total_price, creditcardprice, menuqty_cnt). "
        "Preserve the original capitalization and spelling exactly as written. "
        "Use empty strings for missing values. Do not add any other text or markdown."
    ),
}


def load_key() -> str:
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise SystemExit("Set LLM_API_KEY to your LLM provider API key.")
    return key


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_json_object(text: str) -> dict[str, Any]:
    """Robustly extract the first balanced JSON object from an LLM reply."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                    return obj if isinstance(obj, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def _call_llm(client, model: str, system: str, user: str, max_retries: int, timeout: float):
    """Single LLM call with exponential-backoff retry."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                extra_body={"reasoning_effort": "none"},
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries:
                raise
            wait = min(60.0, attempt * 10.0)
            print(f"  LLM API error; retrying in {wait}s (attempt {attempt}/{max_retries}): {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _extract_one(client, model: str, system: str, pred: dict[str, Any], max_retries: int, timeout: float) -> dict[str, Any]:
    sample_id = pred["sample_id"]
    text = pred.get("text", "")
    start = time.perf_counter()
    try:
        resp = _call_llm(client, model, system, text, max_retries, timeout)
        latency_ms = (time.perf_counter() - start) * 1000
        content = resp.choices[0].message.content
        usage = resp.usage
        return {
            "sample_id": sample_id,
            "dataset_id": pred.get("dataset_id", ""),
            "model": pred.get("model", ""),
            "llm_model": model,
            "fields": _parse_json_object(content),
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": int(usage.prompt_tokens) if usage and usage.prompt_tokens is not None else 0,
            "completion_tokens": int(usage.completion_tokens) if usage and usage.completion_tokens is not None else 0,
            "status": "ok",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "sample_id": sample_id,
            "dataset_id": pred.get("dataset_id", ""),
            "model": pred.get("model", ""),
            "llm_model": model,
            "fields": {},
            "latency_ms": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "status": "error",
            "error": str(exc)[:500],
        }


def _write_ordered(results: dict[str, dict[str, Any]], predictions: list[dict[str, Any]], out_path: Path) -> None:
    ordered = [results[p["sample_id"]] for p in predictions if p["sample_id"] in results]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", choices=("sroie", "cord"), required=True)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai SDK is required: pip install openai")

    api_key = load_key()
    predictions = [p for p in read_jsonl(Path(args.predictions)) if p.get("status") == "ok"]
    if args.limit:
        predictions = predictions[: args.limit]

    out_path = Path(args.out)

    # resume: reuse rows already present in the output file
    results: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for row in read_jsonl(out_path):
            results[row["sample_id"]] = row
    remaining = [p for p in predictions if p["sample_id"] not in results]
    print(f"resume: {len(results)} done, {len(remaining)} remaining (of {len(predictions)} ok predictions)", flush=True)

    if not remaining:
        print(out_path)
        return 0

    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=args.timeout)
    system = SYSTEM_PROMPTS[args.dataset]

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_extract_one, client, args.llm_model, system, pred, args.max_retries, args.timeout): pred["sample_id"]
            for pred in remaining
        }
        for future in as_completed(futures):
            row = future.result()
            results[row["sample_id"]] = row
            completed += 1
            if completed % FLUSH_EVERY == 0 or completed == len(remaining):
                _write_ordered(results, predictions, out_path)
                n_ok = sum(1 for r in results.values() if r["status"] == "ok")
                print(f"  progress: {len(results)}/{len(predictions)} results ({n_ok} ok), flushed", flush=True)

    _write_ordered(results, predictions, out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
