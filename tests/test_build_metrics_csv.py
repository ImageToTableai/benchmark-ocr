from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.build_metrics_csv import main


class BuildMetricsCsvTests(unittest.TestCase):
    def test_structured_target_text_metrics_are_not_applicable_and_fields_are_adapted(self) -> None:
        gt_fields = {
            "header": {"invoice_no": "FV/1/2020"},
            "items": [{"item_desc": "Consulting"}],
            "summary": {"total_gross_worth": "123.00"},
        }
        sample = {
            "sample_id": "invoice-1",
            "dataset_id": "invoices_donut",
            "source_split": "train",
            "gt_text": "invoice_no FV/1/2020 total_gross_worth 123.00",
            "gt_fields": gt_fields,
            "doc_type": "invoice",
        }
        prediction = {
            "run_id": "invoice-test",
            "sample_id": "invoice-1",
            "dataset_id": "invoices_donut",
            "model": "structured-vlm",
            "model_version": "1",
            "text": "raw OCR text should not be scored against structured gt_text",
            "fields": gt_fields,
            "latency_ms": 100.0,
            "status": "ok",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "metrics.csv"
            details = root / "field_details.csv"
            samples.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            arguments = [
                "build_metrics_csv.py",
                "--run-id", "invoice-test",
                "--samples", str(samples),
                "--predictions", str(predictions),
                "--run-tier", "provisional",
                "--out", str(output),
            ]
            with patch.object(sys, "argv", arguments):
                self.assertEqual(main(), 0)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = {row["metric"]: row for row in csv.DictReader(handle)}
            with details.open(encoding="utf-8", newline="") as handle:
                detail_rows = list(csv.DictReader(handle))

        self.assertEqual(rows["cer"]["metric_status"], "not_applicable")
        self.assertEqual(rows["wer"]["metric_status"], "not_applicable")
        self.assertIn("not page-level OCR transcription", rows["cer"]["metric_note"])
        self.assertEqual(rows["native_field_value_accuracy"]["value"], "1.0")
        self.assertEqual(rows["native_document_fields_exact"]["value"], "1.0")
        self.assertEqual(
            {row["field_path"] for row in detail_rows},
            {"header.invoice_no", "summary.total_gross_worth"},
        )

    def test_cord_structured_metrics_cost_and_confidence_interval(self) -> None:
        fields = {
            "menu": [{"nm": "Coffee", "cnt": "1 x", "price": "60.000"}],
            "sub_total": {"subtotal_price": "60.000"},
            "total": {"total_price": "60.000"},
        }
        sample = {
            "sample_id": "cord-1",
            "dataset_id": "cord_v2",
            "source_split": "test",
            "gt_text": "Coffee 60.000",
            "gt_fields": fields,
            "doc_type": "receipt",
        }
        prediction = {
            "run_id": "cord-test",
            "sample_id": "cord-1",
            "dataset_id": "cord_v2",
            "model": "structured-ocr",
            "model_version": "1",
            "text": "Coffee 60.000",
            "fields": fields,
            "latency_ms": 100.0,
            "status": "ok",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples.jsonl"
            predictions = root / "predictions.jsonl"
            performance = root / "performance.json"
            output = root / "metrics.csv"
            details = root / "field_details.csv"
            samples.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            performance.write_text(json.dumps({"run_wall_time_ms": 1000, "measurement_mode": "cold_end_to_end"}), encoding="utf-8")
            arguments = [
                "build_metrics_csv.py",
                "--run-id", "cord-test",
                "--samples", str(samples),
                "--predictions", str(predictions),
                "--performance-file", str(performance),
                "--gpu", "rtx-4090",
                "--gpu-provider", "runpod",
                "--gpu-hourly-usd", "1.5",
                "--price-recorded-at-utc", "2026-08-11T00:00:00Z",
                "--bootstrap-resamples", "10",
                "--out", str(output),
            ]
            with patch.object(sys, "argv", arguments):
                self.assertEqual(main(), 0)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = {row["metric"]: row for row in csv.DictReader(handle)}
            with details.open(encoding="utf-8", newline="") as handle:
                detail_rows = list(csv.DictReader(handle))

        self.assertEqual(rows["native_cord_scalar_field_value_accuracy"]["value"], "1.0")
        self.assertEqual(rows["native_cord_menu_line_item_f1"]["value"], "1.0")
        self.assertEqual(rows["native_cord_menu_nm_f1"]["value"], "1.0")
        self.assertEqual(rows["native_cord_menu_price_f1"]["value"], "1.0")
        self.assertEqual(rows["native_cord_document_exact"]["ci95_low"], "1.0")
        self.assertEqual(rows["cost_per_1000_pages"]["value"], "0.4166666666666667")
        self.assertTrue(any(row["field_group"] == "native_cord_scalar" for row in detail_rows))
        self.assertTrue(any(row["field_group"] == "native_cord_menu_by_position" for row in detail_rows))
        self.assertTrue(all(row["match"] == "True" for row in detail_rows))
