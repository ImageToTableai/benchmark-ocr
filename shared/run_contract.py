"""Shared validation helpers for benchmark samples, predictions, and manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


RUN_TIERS = {"smoke", "provisional", "official"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_metadata(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(samples)
    dataset_ids = sorted({str(row.get("dataset_id", "")).strip() for row in rows if row.get("dataset_id")})
    source_splits = sorted({str(row.get("source_split", "")).strip() for row in rows if row.get("source_split")})
    missing_split_count = sum(1 for row in rows if not str(row.get("source_split", "")).strip())
    return {
        "sample_count": len(rows),
        "dataset_ids": dataset_ids,
        "source_splits": source_splits,
        "missing_source_split_count": missing_split_count,
    }


def validate_sample_contract(
    samples: list[dict[str, Any]],
    run_tier: str,
    expected_split: str | None = None,
) -> dict[str, Any]:
    if run_tier not in RUN_TIERS:
        raise ValueError(f"Unsupported run tier {run_tier!r}; expected one of {sorted(RUN_TIERS)}")
    if not samples:
        raise ValueError("Sample file is empty")

    metadata = sample_metadata(samples)
    sample_ids = [str(row.get("sample_id", "")) for row in samples]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("Every sample requires a non-empty sample_id")
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Sample file contains duplicate sample_ids: {', '.join(duplicate_ids[:5])}")
    if len(metadata["dataset_ids"]) != 1:
        raise ValueError(f"A run must contain one dataset_id, found {metadata['dataset_ids']}")

    if run_tier == "official":
        if expected_split != "test":
            raise ValueError("Official runs require --expected-split test")
        if metadata["missing_source_split_count"]:
            raise ValueError("Official runs require source_split on every sample")

    if expected_split:
        unexpected = [
            row.get("sample_id", "<unknown>")
            for row in samples
            if str(row.get("source_split", "")).strip() != expected_split
        ]
        if unexpected:
            preview = ", ".join(map(str, unexpected[:5]))
            raise ValueError(
                f"Expected source_split={expected_split!r}, found {len(unexpected)} mismatched samples: {preview}"
            )

    return metadata


def validate_prediction_contract(
    predictions: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    run_id: str,
    run_tier: str = "smoke",
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("No predictions found")

    sample_ids = {str(row.get("sample_id", "")) for row in samples}
    unknown_ids = [row.get("sample_id", "<unknown>") for row in predictions if row.get("sample_id") not in sample_ids]
    if unknown_ids:
        preview = ", ".join(map(str, unknown_ids[:5]))
        raise ValueError(f"Predictions reference unknown sample_ids: {preview}")

    wrong_run_ids = [row.get("sample_id", "<unknown>") for row in predictions if row.get("run_id") != run_id]
    if wrong_run_ids:
        preview = ", ".join(map(str, wrong_run_ids[:5]))
        raise ValueError(f"Predictions do not match run_id {run_id!r}: {preview}")

    sample_dataset_id = str(samples[0].get("dataset_id", ""))
    wrong_dataset_ids = [
        row.get("sample_id", "<unknown>")
        for row in predictions
        if str(row.get("dataset_id", "")) != sample_dataset_id
    ]
    if wrong_dataset_ids:
        preview = ", ".join(map(str, wrong_dataset_ids[:5]))
        raise ValueError(f"Predictions do not match dataset_id {sample_dataset_id!r}: {preview}")

    prediction_sample_ids = [str(row.get("sample_id", "")) for row in predictions]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(prediction_sample_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Predictions contain duplicate sample_ids: {', '.join(duplicate_ids[:5])}")

    missing_ids = sorted(sample_ids - set(prediction_sample_ids))
    if run_tier == "official" and missing_ids:
        raise ValueError(f"Official run is missing predictions for {len(missing_ids)} samples: {', '.join(missing_ids[:5])}")

    models = sorted({str(row.get("model", "")) for row in predictions if row.get("model")})
    if len(models) != 1:
        raise ValueError(f"A runner output must contain one model identifier, found {models}")

    return {
        "prediction_count": len(predictions),
        "missing_prediction_count": len(missing_ids),
        "models": models,
        "model_versions": sorted({str(row.get("model_version", "")) for row in predictions if row.get("model_version")}),
    }


def validate_warmup_samples(scored_samples: list[dict[str, Any]], warmup_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject warm-up inputs that overlap the scored fixed test set."""
    validate_sample_contract(warmup_samples, "smoke")
    scored_ids = {str(row["sample_id"]) for row in scored_samples}
    warmup_ids = {str(row["sample_id"]) for row in warmup_samples}
    overlap = sorted(scored_ids & warmup_ids)
    if overlap:
        raise ValueError(f"Warm-up samples overlap scored samples: {', '.join(overlap[:5])}")
    return {"warmup_sample_count": len(warmup_samples)}
