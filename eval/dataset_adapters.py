"""Dataset-specific adapters for structured-document benchmark targets."""

from __future__ import annotations

from typing import Any


STRUCTURED_TARGET_DATASETS = frozenset(
    {
        "fake_w2",
        "mychen76_invoices",
        "invoices_donut",
    }
)

STRUCTURED_TARGET_TEXT_NOTE = (
    "Dataset gt_text is a flattened structured target, not page-level OCR "
    "transcription; use field metrics and dataset adapters."
)


def text_metrics_applicable(dataset_id: str) -> bool:
    return dataset_id not in STRUCTURED_TARGET_DATASETS


def structured_target_note(dataset_id: str) -> str:
    if text_metrics_applicable(dataset_id):
        return ""
    return STRUCTURED_TARGET_TEXT_NOTE


def _scalar_fields(fields: Any) -> dict[str, str]:
    if not isinstance(fields, dict):
        return {}
    return {
        str(key): value
        for key, value in fields.items()
        if value is not None and not isinstance(value, (dict, list, tuple))
    }


def _flatten_mapping(prefix: str, value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    flattened: dict[str, str] = {}
    for key, child in value.items():
        if child is None or isinstance(child, (dict, list, tuple)):
            continue
        flattened[f"{prefix}.{key}"] = child
    return flattened


def adapt_fields(dataset_id: str, fields: Any) -> dict[str, str]:
    """Return a flat, comparable scalar field map for a dataset.

    This deliberately excludes repeated line items for invoice datasets. Line
    items need row-aware metrics; flattening them into scalar keys would hide
    row matching errors and overstate comparability.
    """
    if dataset_id == "invoices_donut":
        if not isinstance(fields, dict):
            return {}
        adapted: dict[str, str] = {}
        adapted.update(_flatten_mapping("header", fields.get("header")))
        adapted.update(_flatten_mapping("summary", fields.get("summary")))
        adapted.update(_scalar_fields(fields))
        return adapted
    if dataset_id in {"fake_w2", "mychen76_invoices"}:
        return _scalar_fields(fields)
    return _scalar_fields(fields)


def adapt_sample(dataset_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(sample)
    adapted["gt_fields"] = adapt_fields(dataset_id, sample.get("gt_fields", {}))
    return adapted
