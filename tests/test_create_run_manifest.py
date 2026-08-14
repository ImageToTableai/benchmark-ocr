from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.create_run_manifest import main
from shared.run_contract import sha256_file


class CreateRunManifestTests(unittest.TestCase):
    def test_manifest_records_performance_hash_and_actual_price_metadata(self) -> None:
        sample = {"sample_id": "a", "dataset_id": "receipt", "source_split": "test"}
        prediction = {"sample_id": "a", "dataset_id": "receipt", "model": "ocr", "run_id": "official-001"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples.jsonl"
            predictions = root / "predictions.jsonl"
            runner = root / "runner.py"
            performance = root / "performance.json"
            output = root / "manifest.json"
            samples.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            runner.write_text("# fixture\n", encoding="utf-8")
            performance.write_text(json.dumps({"run_wall_time_ms": 1234, "measurement_mode": "cold_end_to_end"}), encoding="utf-8")
            arguments = [
                "create_run_manifest.py",
                "--run-id", "official-001",
                "--run-tier", "official",
                "--expected-split", "test",
                "--samples", str(samples),
                "--predictions", str(predictions),
                "--runner", str(runner),
                "--model", "paddleocr",
                "--env-python", "/tmp/env/bin/python",
                "--performance-file", str(performance),
                "--gpu-label", "rtx-4090",
                "--gpu-provider", "runpod",
                "--gpu-hourly-usd", "1.5",
                "--price-recorded-at-utc", "2026-08-11T00:00:00Z",
                "--out", str(output),
            ]
            with patch.object(sys, "argv", arguments):
                self.assertEqual(main(), 0)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            performance_hash = sha256_file(performance)

        self.assertEqual(manifest["performance"]["sha256"], performance_hash)
        self.assertEqual(manifest["performance"]["run_wall_time_ms"], 1234)
        self.assertEqual(manifest["cost"]["gpu_hourly_usd"], 1.5)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["runner"]["model"], "paddleocr")
        self.assertEqual(manifest["runner"]["env_python"], "/tmp/env/bin/python")
        self.assertIn("python_executable", manifest["runtime"])
        self.assertIn("pip_freeze_sha256", manifest["packages"])
        self.assertIn("key_versions", manifest["packages"])
        self.assertIn("git_status_sha256", manifest["source"])
