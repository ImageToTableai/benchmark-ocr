#!/usr/bin/env python3
"""Compare regex vs LLM field postprocessing for one model/dataset run.

Computes LLM field metrics with the same functions the regex postprocessor uses
and reads the regex metrics from the existing metrics.csv, emitting one combined
comparison row.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.field_metrics import document_fields_exact, field_value_accuracy, field_value_f1
from eval.cord_metrics import cord_scalar_fields


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_fields(pred_fields: list[dict[str, Any]], gt_fields: list[dict[str, Any]], dataset: str) -> dict[str, float]:
    if dataset == "cord":
        pred_fields = [cord_scalar_fields(f) for f in pred_fields]
        gt_fields = [cord_scalar_fields(g) for g in gt_fields]
    return {
        "field_value_accuracy": mean([field_value_accuracy(p, g) for p, g in zip(pred_fields, gt_fields, strict=True)]),
        "field_value_f1": mean([field_value_f1(p, g)["f1"] for p, g in zip(pred_fields, gt_fields, strict=True)]),
        "document_fields_exact": mean([document_fields_exact(p, g) for p, g in zip(pred_fields, gt_fields, strict=True)]),
    }


def read_regex_metrics(metrics_csv: Path, dataset: str) -> dict[str, float | None]:
    prefix = "postprocessed_sroie_receipt_regex" if dataset == "sroie" else "postprocessed_cord_scalar"
    result: dict[str, float | None] = {}
    with metrics_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            metric = row.get("metric", "")
            if not metric.startswith(prefix + "_"):
                continue
            key = metric[len(prefix) + 1 :]
            try:
                result[key] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                result[key] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-fields", required=True, help="LLM postprocessor output JSONL")
    parser.add_argument("--samples", required=True, help="GT samples JSONL")
    parser.add_argument("--metrics-csv", required=True, help="Existing metrics.csv with regex scores")
    parser.add_argument("--dataset", choices=("sroie", "cord"), required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--repeat", default="", help="Repeat index for the repeat experiment (empty = single-run comparison)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    llm_rows = [r for r in read_jsonl(Path(args.llm_fields)) if r.get("status") == "ok"]
    samples = read_jsonl(Path(args.samples))
    samples_by_id = {s["sample_id"]: s for s in samples}

    model = llm_rows[0]["model"] if llm_rows else "unknown"
    dataset_id = llm_rows[0]["dataset_id"] if llm_rows else samples[0]["dataset_id"] if samples else "unknown"

    pred_fields = [r.get("fields", {}) for r in llm_rows]
    gt_fields = [samples_by_id.get(r["sample_id"], {}).get("gt_fields", {}) for r in llm_rows]

    llm_scores = score_fields(pred_fields, gt_fields, args.dataset)
    regex_scores = read_regex_metrics(Path(args.metrics_csv), args.dataset)

    # aggregate latency/cost
    latencies = [r["latency_ms"] for r in llm_rows if r.get("latency_ms") is not None]
    prompt_tokens = sum(r.get("prompt_tokens", 0) for r in llm_rows)
    completion_tokens = sum(r.get("completion_tokens", 0) for r in llm_rows)

    out_row: dict[str, Any] = {
        "model": model,
        "dataset": dataset_id,
        "llm_model": args.llm_model,
        "regex_field_value_accuracy": regex_scores.get("field_value_accuracy"),
        "llm_field_value_accuracy": llm_scores["field_value_accuracy"],
        "regex_field_value_f1": regex_scores.get("field_value_f1"),
        "llm_field_value_f1": llm_scores["field_value_f1"],
        "regex_document_fields_exact": regex_scores.get("document_fields_exact"),
        "llm_document_fields_exact": llm_scores["document_fields_exact"],
        "llm_ok_count": len(llm_rows),
        "llm_median_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "llm_prompt_tokens": prompt_tokens,
        "llm_completion_tokens": completion_tokens,
    }
    if args.repeat:
        out_row = {"repeat": args.repeat, **out_row}

    out_path = Path(args.out) if args.out else Path("results") / "field_method_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(out_row)

    print(json.dumps(out_row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
