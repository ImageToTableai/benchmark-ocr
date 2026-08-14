"""CORD v2 nested receipt-field normalization and scoring helpers."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


MENU_KEYS = ("nm", "num", "cnt", "price", "itemsubtotal")
AMOUNT_SUFFIXES = ("price", "subtotal", "discount", "tax", "etc")
COUNT_SUFFIXES = ("cnt",)


def _text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _normalize_amount(value: Any) -> str:
    text = _text(value)
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"-?[0-9]{1,3}(?:[,.][0-9]{3})+", compact):
        return re.sub(r"[,.]", "", compact)
    if re.fullmatch(r"-?[0-9]+(?:[,.]0+)?", compact):
        integer, separator, fraction = compact.replace(",", ".").partition(".")
        if separator and set(fraction) == {"0"}:
            return integer
        return compact
    return text


def _normalize_count(value: Any) -> str:
    text = re.sub(r"\s*x$", "", _text(value))
    if re.fullmatch(r"-?[0-9]+(?:[\.,][0-9]+)?", text):
        integer, separator, fraction = text.replace(",", ".").partition(".")
        if separator and set(fraction) == {"0"}:
            return integer
    return text


def normalize_cord_value(path: str, value: Any) -> str:
    """Normalize CORD text while preserving its distinction between amounts and names."""
    leaf = path.rsplit(".", 1)[-1]
    if leaf.endswith(COUNT_SUFFIXES):
        return _normalize_count(value)
    if leaf.endswith(AMOUNT_SUFFIXES):
        return _normalize_amount(value)
    return _text(value)


def cord_scalar_fields(fields: Any) -> dict[str, str]:
    """Flatten CORD subtotal/total scalars while deliberately excluding line items."""
    if not isinstance(fields, dict):
        return {}

    result: dict[str, str] = {}
    for section in ("sub_total", "total"):
        value = fields.get(section)
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(item, (dict, list, tuple)):
                    path = f"{section}.{key}"
                    result[path] = normalize_cord_value(path, item)

    # Structured models may already emit CORD's dotted field paths.
    for key, value in fields.items():
        if isinstance(key, str) and "." in key and not isinstance(value, (dict, list, tuple)):
            result[key] = normalize_cord_value(key, value)
    return result


def has_cord_structured_fields(fields: Any) -> bool:
    """Return whether a prediction exposes a CORD-compatible scalar or menu shape."""
    return bool(cord_scalar_fields(fields)) or (
        isinstance(fields, dict) and isinstance(fields.get("menu"), (dict, list))
    )


def _menu_rows(fields: Any) -> list[dict[str, str]]:
    if not isinstance(fields, dict):
        return []
    menu = fields.get("menu")
    if isinstance(menu, dict):
        items = [menu]
    elif isinstance(menu, list):
        items = menu
    else:
        return []

    rows: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = {
            key: normalize_cord_value(f"menu.{key}", value)
            for key, value in item.items()
            if key in MENU_KEYS and not isinstance(value, (dict, list, tuple)) and _text(value)
        }
        if row:
            rows.append(row)
    return rows


def cord_menu_column_scores(predicted_fields: Any, ground_truth_fields: Any) -> dict[str, dict[str, float]]:
    """Column-wise CORD menu diagnostics using duplicate-aware normalized values.

    This intentionally ignores row association. Use it to diagnose whether a
    model recovered names, counts, prices, or subtotals; keep
    `cord_menu_line_item_scores` as the row-exact line-item metric.
    """
    predicted_rows = _menu_rows(predicted_fields)
    ground_truth_rows = _menu_rows(ground_truth_fields)
    scores: dict[str, dict[str, float]] = {}
    for key in MENU_KEYS:
        predicted_values = [row[key] for row in predicted_rows if row.get(key)]
        ground_truth_values = [row[key] for row in ground_truth_rows if row.get(key)]
        predicted_counts = Counter(predicted_values)
        ground_truth_counts = Counter(ground_truth_values)
        matched = sum(min(predicted_counts[value], ground_truth_counts[value]) for value in predicted_counts.keys() | ground_truth_counts.keys())
        predicted_count = sum(predicted_counts.values())
        ground_truth_count = sum(ground_truth_counts.values())
        precision = matched / predicted_count if predicted_count else 0.0
        recall = matched / ground_truth_count if ground_truth_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[key] = {
            "matched": float(matched),
            "predicted_count": float(predicted_count),
            "ground_truth_count": float(ground_truth_count),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return scores


def _items_match(predicted: dict[str, str], ground_truth: dict[str, str]) -> bool:
    return bool(ground_truth) and all(predicted.get(key) == value for key, value in ground_truth.items())


def _maximum_menu_matches(predicted: list[dict[str, str]], ground_truth: list[dict[str, str]]) -> int:
    """Maximum bipartite match prevents duplicate menu entries from being overcounted."""
    matched_prediction_for_gt = [-1] * len(ground_truth)

    def visit(prediction_index: int, seen: set[int]) -> bool:
        for ground_truth_index, ground_truth_item in enumerate(ground_truth):
            if ground_truth_index in seen or not _items_match(predicted[prediction_index], ground_truth_item):
                continue
            seen.add(ground_truth_index)
            previous_prediction = matched_prediction_for_gt[ground_truth_index]
            if previous_prediction == -1 or visit(previous_prediction, seen):
                matched_prediction_for_gt[ground_truth_index] = prediction_index
                return True
        return False

    return sum(visit(index, set()) for index in range(len(predicted)))


def cord_menu_line_item_scores(predicted_fields: Any, ground_truth_fields: Any) -> dict[str, float]:
    """Micro precision/recall/F1 for complete CORD menu rows, independent of row order."""
    predicted = _menu_rows(predicted_fields)
    ground_truth = _menu_rows(ground_truth_fields)
    matched = _maximum_menu_matches(predicted, ground_truth)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matched": float(matched),
        "predicted_count": float(len(predicted)),
        "ground_truth_count": float(len(ground_truth)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cord_document_exact(predicted_fields: Any, ground_truth_fields: Any) -> float:
    """Require every annotated scalar and menu row to be recovered for one receipt."""
    predicted_scalars = cord_scalar_fields(predicted_fields)
    ground_truth_scalars = cord_scalar_fields(ground_truth_fields)
    scalar_exact = bool(ground_truth_scalars) and all(
        predicted_scalars.get(key) == value for key, value in ground_truth_scalars.items()
    )
    menu_scores = cord_menu_line_item_scores(predicted_fields, ground_truth_fields)
    menu_exact = menu_scores["matched"] == menu_scores["ground_truth_count"] == menu_scores["predicted_count"]
    return float(scalar_exact and menu_exact)
