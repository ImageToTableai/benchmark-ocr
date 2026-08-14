#!/usr/bin/env python3
"""Build metrics.csv with run validation, dataset-aware fields, and evidence metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.confidence_intervals import bootstrap_mean_ci, quantile
from eval.cord_metrics import (
    MENU_KEYS,
    cord_document_exact,
    cord_menu_column_scores,
    cord_menu_line_item_scores,
    cord_scalar_fields,
    has_cord_structured_fields,
    normalize_cord_value,
)
from eval.cord_receipt_regex import extract_cord_fields
from eval.dataset_adapters import adapt_fields, adapt_sample, structured_target_note, text_metrics_applicable
from eval.field_metrics import (
    document_fields_exact,
    extract_demo_fields,
    field_value_accuracy,
    field_value_f1,
    normalize_field_value,
)
from eval.text_metrics import cer, wer
from shared.run_contract import read_jsonl, sha256_file, validate_prediction_contract, validate_sample_contract


FIELD_METRICS = (
    "field_value_accuracy",
    "document_fields_exact",
    "field_value_f1_precision",
    "field_value_f1_recall",
    "field_value_f1",
)
CORD_MENU_METRICS = ("menu_line_item_precision", "menu_line_item_recall", "menu_line_item_f1", "document_exact")
FIELD_DETAIL_COLUMNS = (
    "run_id",
    "model",
    "dataset_id",
    "doc_type",
    "sample_id",
    "field_group",
    "field_path",
    "ground_truth_value",
    "predicted_value",
    "ground_truth_normalized",
    "predicted_normalized",
    "match",
    "status",
    "note",
)


def metric_row(
    run_id: str,
    model: str,
    dataset_id: str,
    doc_type: str,
    metric: str,
    value: float | None,
    sample_count: int,
    ok_count: int,
    error_count: int,
    metric_status: str = "ok",
    metric_note: str = "",
    ci95_low: float | None = None,
    ci95_high: float | None = None,
    ci95_method: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model": model,
        "dataset_id": dataset_id,
        "doc_type": doc_type,
        "metric": metric,
        "value": value,
        "sample_count": sample_count,
        "ok_count": ok_count,
        "error_count": error_count,
        "metric_status": metric_status,
        "metric_note": metric_note,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "ci95_method": ci95_method,
    }


def flat_field_schema(fields: Any) -> bool:
    return isinstance(fields, dict) and all(not isinstance(value, (dict, list, tuple)) for value in fields.values())


def detail_row(
    context: dict[str, Any],
    sample_id: str,
    field_group: str,
    field_path: str,
    ground_truth_value: Any,
    predicted_value: Any,
    ground_truth_normalized: str,
    predicted_normalized: str,
    status: str = "ok",
    note: str = "",
) -> dict[str, Any]:
    return {
        "run_id": context["run_id"],
        "model": context["model"],
        "dataset_id": context["dataset_id"],
        "doc_type": context["doc_type"],
        "sample_id": sample_id,
        "field_group": field_group,
        "field_path": field_path,
        "ground_truth_value": ground_truth_value,
        "predicted_value": predicted_value,
        "ground_truth_normalized": ground_truth_normalized,
        "predicted_normalized": predicted_normalized,
        "match": ground_truth_normalized == predicted_normalized,
        "status": status,
        "note": note,
    }


def add_flat_field_detail_rows(
    detail_rows: list[dict[str, Any]],
    prefix: str,
    fields_by_prediction: list[dict[str, Any]],
    samples_by_prediction: list[dict[str, Any]],
    context: dict[str, Any],
    unavailable_note: str | None = None,
) -> None:
    if unavailable_note:
        return
    for fields, sample in zip(fields_by_prediction, samples_by_prediction, strict=True):
        ground_truth = sample.get("gt_fields", {})
        if not isinstance(ground_truth, dict):
            continue
        for key, gt_value in sorted(ground_truth.items()):
            if isinstance(gt_value, (dict, list, tuple)):
                continue
            pred_value = fields.get(key, "") if isinstance(fields, dict) else ""
            detail_rows.append(
                detail_row(
                    context,
                    str(sample["sample_id"]),
                    prefix,
                    str(key),
                    gt_value,
                    pred_value,
                    normalize_field_value(gt_value),
                    normalize_field_value(pred_value),
                )
            )


def add_cord_field_detail_rows(
    detail_rows: list[dict[str, Any]],
    prefix: str,
    fields_by_prediction: list[dict[str, Any]],
    samples_by_prediction: list[dict[str, Any]],
    context: dict[str, Any],
    unavailable_note: str | None = None,
) -> None:
    if unavailable_note:
        return
    for fields, sample in zip(fields_by_prediction, samples_by_prediction, strict=True):
        sample_id = str(sample["sample_id"])
        ground_truth_fields = sample.get("gt_fields", {})
        gt_scalars = cord_scalar_fields(ground_truth_fields)
        pred_scalars = cord_scalar_fields(fields)
        for path, gt_value in sorted(gt_scalars.items()):
            pred_value = pred_scalars.get(path, "")
            detail_rows.append(
                detail_row(
                    context,
                    sample_id,
                    f"{prefix}_scalar",
                    path,
                    gt_value,
                    pred_value,
                    normalize_cord_value(path, gt_value),
                    normalize_cord_value(path, pred_value),
                )
            )

        gt_menu = ground_truth_fields.get("menu") if isinstance(ground_truth_fields, dict) else None
        pred_menu = fields.get("menu") if isinstance(fields, dict) else None
        gt_rows = gt_menu if isinstance(gt_menu, list) else ([gt_menu] if isinstance(gt_menu, dict) else [])
        pred_rows = pred_menu if isinstance(pred_menu, list) else ([pred_menu] if isinstance(pred_menu, dict) else [])
        for index, gt_row in enumerate(gt_rows):
            if not isinstance(gt_row, dict):
                continue
            pred_row = pred_rows[index] if index < len(pred_rows) and isinstance(pred_rows[index], dict) else {}
            for key in MENU_KEYS:
                if key not in gt_row:
                    continue
                path = f"menu[{index}].{key}"
                pred_value = pred_row.get(key, "")
                detail_rows.append(
                    detail_row(
                        context,
                        sample_id,
                        f"{prefix}_menu_by_position",
                        path,
                        gt_row.get(key, ""),
                        pred_value,
                        normalize_cord_value(f"menu.{key}", gt_row.get(key, "")),
                        normalize_cord_value(f"menu.{key}", pred_value),
                        note="Position-based diagnostic only; row-exact metric is order-independent",
                    )
                )


def add_field_rows(
    rows: list[dict[str, Any]],
    prefix: str,
    fields_by_prediction: list[dict[str, Any]],
    samples_by_prediction: list[dict[str, Any]],
    context: dict[str, Any],
    ci_values: dict[str, list[float]],
    unavailable_note: str | None = None,
) -> None:
    if unavailable_note:
        for metric in FIELD_METRICS:
            rows.append(metric_row(metric=f"{prefix}_{metric}", value=None, metric_status="not_applicable", metric_note=unavailable_note, **context))
        return

    accuracy_values: list[float] = []
    document_exact_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for fields, sample in zip(fields_by_prediction, samples_by_prediction, strict=True):
        ground_truth = sample.get("gt_fields", {})
        accuracy_values.append(field_value_accuracy(fields, ground_truth))
        document_exact_values.append(document_fields_exact(fields, ground_truth))
        f1_result = field_value_f1(fields, ground_truth)
        precision_values.append(f1_result["precision"])
        recall_values.append(f1_result["recall"])
        f1_values.append(f1_result["f1"])

    values = {
        "field_value_accuracy": accuracy_values,
        "document_fields_exact": document_exact_values,
        "field_value_f1_precision": precision_values,
        "field_value_f1_recall": recall_values,
        "field_value_f1": f1_values,
    }
    for metric, document_values in values.items():
        metric_name = f"{prefix}_{metric}"
        ci_values[metric_name] = document_values
        rows.append(metric_row(metric=metric_name, value=sum(document_values) / len(document_values), **context))


def add_cord_native_field_rows(
    rows: list[dict[str, Any]],
    native_fields: list[dict[str, Any]],
    matched_samples: list[dict[str, Any]],
    context: dict[str, Any],
    ci_values: dict[str, list[float]],
    unavailable_note: str | None,
    prefix: str = "native_cord",
) -> None:
    scalar_predictions = [cord_scalar_fields(fields) for fields in native_fields]
    scalar_samples = [{**sample, "gt_fields": cord_scalar_fields(sample.get("gt_fields", {}))} for sample in matched_samples]
    add_field_rows(
        rows,
        f"{prefix}_scalar",
        scalar_predictions,
        scalar_samples,
        context,
        ci_values,
        unavailable_note,
    )
    if unavailable_note:
        for metric in CORD_MENU_METRICS:
            rows.append(metric_row(metric=f"{prefix}_{metric}", value=None, metric_status="not_applicable", metric_note=unavailable_note, **context))
        return

    menu_scores = [cord_menu_line_item_scores(fields, sample.get("gt_fields", {})) for fields, sample in zip(native_fields, matched_samples, strict=True)]
    total_matched = sum(score["matched"] for score in menu_scores)
    total_predicted = sum(score["predicted_count"] for score in menu_scores)
    total_ground_truth = sum(score["ground_truth_count"] for score in menu_scores)
    precision = total_matched / total_predicted if total_predicted else 0.0
    recall = total_matched / total_ground_truth if total_ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    menu_values = {
        "menu_line_item_precision": precision,
        "menu_line_item_recall": recall,
        "menu_line_item_f1": f1,
    }
    for metric, value in menu_values.items():
        rows.append(
            metric_row(
                metric=f"{prefix}_{metric}",
                value=value,
                metric_note="Corpus micro score over complete, order-independent CORD menu rows",
                **context,
            )
        )

    column_scores = cord_menu_column_scores(
        {"menu": [row for fields in native_fields for row in (fields.get("menu", []) if isinstance(fields.get("menu"), list) else ([fields.get("menu")] if isinstance(fields.get("menu"), dict) else [])) if isinstance(row, dict)]},
        {"menu": [row for sample in matched_samples for row in (sample.get("gt_fields", {}).get("menu", []) if isinstance(sample.get("gt_fields", {}).get("menu"), list) else ([sample.get("gt_fields", {}).get("menu")] if isinstance(sample.get("gt_fields", {}).get("menu"), dict) else [])) if isinstance(row, dict)]},
    )
    for key, scores in column_scores.items():
        for metric in ("precision", "recall", "f1"):
            rows.append(
                metric_row(
                    metric=f"{prefix}_menu_{key}_{metric}",
                    value=scores[metric],
                    metric_note="Corpus micro column diagnostic over normalized CORD menu values; ignores row association",
                    **context,
                )
            )

    document_exact_values = [
        cord_document_exact(fields, sample.get("gt_fields", {}))
        for fields, sample in zip(native_fields, matched_samples, strict=True)
    ]
    ci_values[f"{prefix}_document_exact"] = document_exact_values
    rows.append(metric_row(metric=f"{prefix}_document_exact", value=sum(document_exact_values) / len(document_exact_values), **context))


def load_performance(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    performance_path = Path(path)
    try:
        value = json.loads(performance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read performance file {performance_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Performance file {performance_path} must contain an object")
    return value


def validate_manifest(path: Path, args: argparse.Namespace, samples_path: Path) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read manifest {path}: {exc}") from exc
    if manifest.get("run_id") != args.run_id or manifest.get("run_tier") != args.run_tier:
        raise SystemExit("Manifest run_id/run_tier does not match evaluation arguments")
    if manifest.get("data", {}).get("samples_sha256") != sha256_file(samples_path):
        raise SystemExit("Manifest sample hash does not match --samples")
    if args.performance_file and manifest.get("performance", {}).get("sha256") != sha256_file(Path(args.performance_file)):
        raise SystemExit("Manifest performance hash does not match --performance-file")


def attach_confidence_intervals(
    rows: list[dict[str, Any]],
    values_by_metric: dict[str, list[float]],
    resamples: int,
    seed: int,
) -> None:
    method = f"percentile_bootstrap_95;resamples={resamples};seed={seed}"
    for row in rows:
        values = values_by_metric.get(row["metric"])
        if values is None:
            continue
        interval = bootstrap_mean_ci(values, resamples=resamples, seed=seed)
        if interval is not None:
            row["ci95_low"], row["ci95_high"] = interval
            row["ci95_method"] = method


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--run-tier", choices=("smoke", "provisional", "official"), default="smoke")
    parser.add_argument("--expected-split", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--field-postprocessor",
        choices=("none", "sroie_receipt_regex", "cord_receipt_regex"),
        default="none",
        help="Optional named postprocessor; its scores remain separate from native model fields.",
    )
    parser.add_argument("--performance-file", default=None, help="JSON metadata emitted by server/run_model.sh")
    parser.add_argument("--gpu", default=None, help="GPU label recorded with a cost result; never implies a catalog price")
    parser.add_argument("--gpu-provider", default=None)
    parser.add_argument("--gpu-hourly-usd", type=float, default=None)
    parser.add_argument("--price-recorded-at-utc", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--field-details-out",
        default=None,
        help="Optional per-field diagnostic CSV path. Defaults to field_details.csv beside metrics output.",
    )
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        raise SystemExit("--bootstrap-resamples must be positive")
    if args.gpu_hourly_usd is not None and args.gpu_hourly_usd < 0:
        raise SystemExit("--gpu-hourly-usd must be non-negative")

    samples_path = Path(args.samples)
    predictions_path = Path(args.predictions)
    samples_list = read_jsonl(samples_path)
    predictions = read_jsonl(predictions_path)
    validate_sample_contract(samples_list, args.run_tier, args.expected_split)
    validate_prediction_contract(predictions, samples_list, args.run_id, args.run_tier)
    if args.run_tier == "official":
        if not args.manifest:
            raise SystemExit("Official runs require --manifest generated by scripts/create_run_manifest.py")
        validate_manifest(Path(args.manifest), args, samples_path)

    performance = load_performance(args.performance_file)
    samples = {row["sample_id"]: row for row in samples_list}
    ok_predictions = [row for row in predictions if row.get("status") == "ok"]
    sample_count = len(predictions)
    ok_count = len(ok_predictions)
    error_count = sample_count - ok_count
    first_pred = predictions[0]
    model = first_pred["model"]
    dataset_id = first_pred["dataset_id"]
    doc_type = samples[first_pred["sample_id"]].get("doc_type", "")
    context = {
        "run_id": args.run_id,
        "model": model,
        "dataset_id": dataset_id,
        "doc_type": doc_type,
        "sample_count": sample_count,
        "ok_count": ok_count,
        "error_count": error_count,
    }

    matched_samples = [samples[pred["sample_id"]] for pred in ok_predictions]
    text_note = structured_target_note(dataset_id)
    if text_metrics_applicable(dataset_id):
        cer_values = [cer(pred.get("text", ""), sample.get("gt_text", "")) for pred, sample in zip(ok_predictions, matched_samples, strict=True)]
        wer_values = [wer(pred.get("text", ""), sample.get("gt_text", "")) for pred, sample in zip(ok_predictions, matched_samples, strict=True)]
    else:
        cer_values = []
        wer_values = []
    latencies = [float(pred["latency_ms"]) for pred in ok_predictions if pred.get("latency_ms") is not None]
    ci_values: dict[str, list[float]] = {
        "error_rate": [float(pred.get("status") != "ok") for pred in predictions],
        "success_rate": [float(pred.get("status") == "ok") for pred in predictions],
    }
    if text_metrics_applicable(dataset_id):
        ci_values["cer"] = cer_values
        ci_values["wer"] = wer_values

    text_metric_status = "ok" if text_metrics_applicable(dataset_id) else "not_applicable"
    rows = [
        metric_row(metric="cer", value=sum(cer_values) / len(cer_values) if cer_values else None, metric_status=text_metric_status, metric_note=text_note, **context),
        metric_row(metric="wer", value=sum(wer_values) / len(wer_values) if wer_values else None, metric_status=text_metric_status, metric_note=text_note, **context),
        metric_row(metric="latency_p50_ms", value=quantile(latencies, 0.50) if latencies else None, metric_note="Per-successful-prediction latency; excludes runner initialization", **context),
        metric_row(metric="latency_p95_ms", value=quantile(latencies, 0.95) if latencies else None, metric_note="Per-successful-prediction latency; excludes runner initialization", **context),
        metric_row(metric="error_rate", value=error_count / sample_count, **context),
        metric_row(metric="success_rate", value=ok_count / sample_count, **context),
    ]

    if latencies:
        average_latency_ms = sum(latencies) / len(latencies)
        rows.append(metric_row(metric="pages_per_minute", value=60000.0 / average_latency_ms, metric_note="Uses summed successful per-page latency; excludes runner initialization", **context))
    else:
        rows.append(metric_row(metric="pages_per_minute", value=None, metric_status="not_applicable", metric_note="No successful prediction latency recorded", **context))

    wall_time_ms: float | None = None
    if performance is not None:
        try:
            wall_time_ms = float(performance["run_wall_time_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("Performance file requires numeric run_wall_time_ms") from exc
        if wall_time_ms <= 0:
            raise SystemExit("Performance run_wall_time_ms must be positive")
        measurement_mode = str(performance.get("measurement_mode", "unspecified"))
        note = f"measurement_mode={measurement_mode}; includes runner process initialization"
        rows.append(metric_row(metric="run_wall_time_ms", value=wall_time_ms, metric_note=note, **context))
        rows.append(
            metric_row(
                metric="wall_clock_pages_per_minute",
                value=ok_count * 60000.0 / wall_time_ms,
                metric_note=note,
                **context,
            )
        )
    else:
        rows.append(metric_row(metric="run_wall_time_ms", value=None, metric_status="not_applicable", metric_note="No performance file supplied", **context))
        rows.append(metric_row(metric="wall_clock_pages_per_minute", value=None, metric_status="not_applicable", metric_note="No performance file supplied", **context))

    if args.gpu_hourly_usd is not None and wall_time_ms is not None and ok_count:
        if not args.gpu_provider or not args.price_recorded_at_utc:
            raise SystemExit("Cost requires --gpu-provider and --price-recorded-at-utc with --gpu-hourly-usd")
        cost_per_1000 = args.gpu_hourly_usd * wall_time_ms / 3600000.0 / ok_count * 1000
        cost_note = (
            f"GPU={args.gpu or 'unspecified'}; provider={args.gpu_provider}; "
            f"hourly_usd={args.gpu_hourly_usd}; price_recorded_at_utc={args.price_recorded_at_utc}; uses run wall time"
        )
        rows.append(metric_row(metric="cost_per_1000_pages", value=cost_per_1000, metric_note=cost_note, **context))
    else:
        rows.append(metric_row(metric="cost_per_1000_pages", value=None, metric_status="not_applicable", metric_note="Requires actual hourly USD price and run_wall_time_ms", **context))

    adapted_matched_samples = [adapt_sample(dataset_id, sample) for sample in matched_samples]
    ground_truth_fields = [sample.get("gt_fields", {}) for sample in adapted_matched_samples]
    native_fields = [adapt_fields(dataset_id, pred.get("fields") if isinstance(pred.get("fields"), dict) else {}) for pred in ok_predictions]
    field_detail_rows: list[dict[str, Any]] = []
    if dataset_id == "cord_v2":
        ground_truth_fields = [sample.get("gt_fields", {}) for sample in matched_samples]
        native_fields = [pred.get("fields") if isinstance(pred.get("fields"), dict) else {} for pred in ok_predictions]
        field_unavailable = None
        if not any(ground_truth_fields):
            field_unavailable = "Dataset has no field ground truth"
        elif not any(has_cord_structured_fields(fields) for fields in native_fields):
            field_unavailable = "Model emitted no native CORD structured fields"
        add_cord_native_field_rows(rows, native_fields, matched_samples, context, ci_values, field_unavailable)
        add_cord_field_detail_rows(field_detail_rows, "native_cord", native_fields, matched_samples, context, field_unavailable)
    else:
        if not any(ground_truth_fields):
            field_unavailable = "Dataset has no field ground truth"
        elif not all(flat_field_schema(fields) for fields in ground_truth_fields):
            field_unavailable = "Dataset field schema is nested; add a dataset-specific field adapter before scoring"
        else:
            field_unavailable = None
        native_unavailable = field_unavailable
        if native_unavailable is None and not any(native_fields):
            native_unavailable = "Model emitted no native structured fields"
        add_field_rows(rows, "native", native_fields, adapted_matched_samples, context, ci_values, native_unavailable)
        add_flat_field_detail_rows(field_detail_rows, "native", native_fields, adapted_matched_samples, context, native_unavailable)

    if args.field_postprocessor == "none":
        add_field_rows(rows, "postprocessed", [], [], context, ci_values, "No named field postprocessor selected")
    elif args.field_postprocessor == "sroie_receipt_regex":
        if dataset_id != "sroie_2019":
            add_field_rows(rows, "postprocessed", [], [], context, ci_values, "sroie_receipt_regex is only valid for sroie_2019")
        else:
            postprocessed_fields = [extract_demo_fields(pred.get("text", "")) for pred in ok_predictions]
            add_field_rows(rows, "postprocessed_sroie_receipt_regex", postprocessed_fields, matched_samples, context, ci_values, field_unavailable)
            add_flat_field_detail_rows(
                field_detail_rows,
                "postprocessed_sroie_receipt_regex",
                postprocessed_fields,
                matched_samples,
                context,
                field_unavailable,
            )

    if args.field_postprocessor == "cord_receipt_regex":
        if dataset_id != "cord_v2":
            add_cord_native_field_rows(rows, [], [], context, ci_values, "cord_receipt_regex is only valid for cord_v2", prefix="postprocessed_cord")
        else:
            postprocessed_cord_fields = [extract_cord_fields(pred.get("text", "")) for pred in ok_predictions]
            cord_post_unavailable = None if any(ground_truth_fields) else "Dataset has no field ground truth"
            add_cord_native_field_rows(rows, postprocessed_cord_fields, matched_samples, context, ci_values, cord_post_unavailable, prefix="postprocessed_cord")
            add_cord_field_detail_rows(
                field_detail_rows,
                "postprocessed_cord",
                postprocessed_cord_fields,
                matched_samples,
                context,
                cord_post_unavailable,
            )

    attach_confidence_intervals(rows, ci_values, args.bootstrap_resamples, args.bootstrap_seed)
    out_path = Path(args.out) if args.out else Path("results") / args.run_id / "metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    details_path = Path(args.field_details_out) if args.field_details_out else out_path.parent / "field_details.csv"
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELD_DETAIL_COLUMNS))
        writer.writeheader()
        writer.writerows(field_detail_rows)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
