from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_dataset_samples import audit_samples


class AuditDatasetSamplesTests(unittest.TestCase):
    def test_reports_test_readiness_and_nested_field_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"fixture")
            samples = root / "samples.jsonl"
            samples.write_text(
                json.dumps(
                    {
                        "sample_id": "a",
                        "dataset_id": "receipt",
                        "source_repo": "example/source",
                        "source_split": "test",
                        "image_path": str(image),
                        "gt_text": "TOTAL 10",
                        "gt_fields": {"total": {"price": "10"}},
                        "doc_type": "receipt",
                        "language": "en",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = audit_samples(samples)

        self.assertTrue(report["technical_test_ready"])
        self.assertEqual(report["ground_truth_field_shapes"], {"nested_object": 1})
