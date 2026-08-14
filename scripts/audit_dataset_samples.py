#!/usr/bin/env python3
"""Audit one normalized sample JSONL before it is used for a benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.run_contract import read_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field_shape(value: Any) -> str:
    if not value:
        return "empty"
    if isinstance(value, list):
        return "list"
    if not isinstance(value, dict):
        return type(value).__name__
    if any(isinstance(item, (dict, list, tuple)) for item in value.values()):
        return "nested_object"
    return "flat_object"


def audit_samples(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    image_paths = [Path(str(row.get("image_path", ""))) for row in rows]
    report = {
        "schema_version": 1,
        "sample_file": str(path),
        "samples_sha256": sha256_file(path),
        "sample_count": len(rows),
        "dataset_ids": sorted({str(row.get("dataset_id", "")) for row in rows}),
        "source_repos": sorted({str(row.get("source_repo", "")) for row in rows}),
        "source_configs": sorted({str(row.get("source_config", "")) for row in rows}),
        "source_splits": dict(sorted(Counter(str(row.get("source_split", "")) for row in rows).items())),
        "doc_types": dict(sorted(Counter(str(row.get("doc_type", "")) for row in rows).items())),
        "languages": dict(sorted(Counter(str(row.get("language", "")) for row in rows).items())),
        "missing_image_count": sum(not image_path.is_file() for image_path in image_paths),
        "ground_truth_text_count": sum(bool(str(row.get("gt_text", "")).strip()) for row in rows),
        "ground_truth_fields_count": sum(bool(row.get("gt_fields")) for row in rows),
        "ground_truth_field_shapes": dict(sorted(Counter(field_shape(row.get("gt_fields")) for row in rows).items())),
    }
    report["technical_test_ready"] = (
        report["sample_count"] > 0
        and report["source_splits"] == {"test": report["sample_count"]}
        and report["missing_image_count"] == 0
        and report["ground_truth_text_count"] == report["sample_count"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = audit_samples(Path(args.samples))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
