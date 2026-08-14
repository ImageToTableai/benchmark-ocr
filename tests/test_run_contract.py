from __future__ import annotations

import unittest

from shared.run_contract import validate_prediction_contract, validate_sample_contract, validate_warmup_samples


class RunContractTests(unittest.TestCase):
    def test_official_run_accepts_test_split(self) -> None:
        result = validate_sample_contract(
            [{"sample_id": "a", "dataset_id": "receipt", "source_split": "test"}],
            "official",
            "test",
        )
        self.assertEqual(result["source_splits"], ["test"])

    def test_official_run_rejects_train_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "mismatched"):
            validate_sample_contract(
                [{"sample_id": "a", "dataset_id": "receipt", "source_split": "train"}],
                "official",
                "test",
            )

    def test_smoke_run_allows_legacy_sample_without_split(self) -> None:
        result = validate_sample_contract(
            [{"sample_id": "a", "dataset_id": "local_demo"}],
            "smoke",
        )
        self.assertEqual(result["missing_source_split_count"], 1)

    def test_official_run_rejects_missing_prediction(self) -> None:
        samples = [
            {"sample_id": "a", "dataset_id": "receipt", "source_split": "test"},
            {"sample_id": "b", "dataset_id": "receipt", "source_split": "test"},
        ]
        predictions = [{"sample_id": "a", "dataset_id": "receipt", "model": "ocr", "run_id": "official-001"}]
        with self.assertRaisesRegex(ValueError, "missing predictions"):
            validate_prediction_contract(predictions, samples, "official-001", "official")

    def test_warmup_samples_must_not_overlap_scored_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_warmup_samples(
                [{"sample_id": "a", "dataset_id": "receipt"}],
                [{"sample_id": "a", "dataset_id": "receipt"}],
            )
