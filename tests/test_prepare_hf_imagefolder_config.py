from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_hf_imagefolder_config import download_config


class PrepareHfImagefolderConfigTests(unittest.TestCase):
    def test_writes_normalized_samples_and_provenance(self) -> None:
        records = [
            {
                "file_name": "page one.png",
                "doc_id": "document-1",
                "text": "Xin chao",
                "license": "CC0-1.0",
                "gen_method": "scan_artifacts",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("scripts.prepare_hf_imagefolder_config.load_metadata", return_value=records), patch(
                "scripts.prepare_hf_imagefolder_config.fetch_bytes", return_value=b"image-bytes"
            ):
                written = download_config(
                    dataset_id="vietnamese_ocr",
                    repo="owner/repo",
                    revision="abc123",
                    config="receipt",
                    limit=0,
                    samples_out=root / "samples.jsonl",
                    image_dir=root / "images",
                    doc_type="receipt",
                    language="vi",
                    timeout=10,
                )

            self.assertEqual(written, 1)
            sample = json.loads((root / "samples.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(sample["source_split"], "test")
            self.assertEqual(sample["source_revision"], "abc123")
            self.assertEqual(sample["source_record_license"], "CC0-1.0")
            self.assertEqual(sample["gt_text"], "Xin chao")
            self.assertTrue(Path(sample["image_path"]).exists())

    def test_failed_download_keeps_existing_output(self) -> None:
        records = [
            {"file_name": "one.png", "doc_id": "one", "text": "one"},
            {"file_name": "two.png", "doc_id": "two", "text": "two"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples_out = root / "samples.jsonl"
            samples_out.write_text("existing\n", encoding="utf-8")
            with patch("scripts.prepare_hf_imagefolder_config.load_metadata", return_value=records), patch(
                "scripts.prepare_hf_imagefolder_config.fetch_bytes", side_effect=[b"first", OSError("network")]
            ):
                with self.assertRaises(OSError):
                    download_config(
                        dataset_id="test",
                        repo="owner/repo",
                        revision="main",
                        config="receipt",
                        limit=0,
                        samples_out=samples_out,
                        image_dir=root / "images",
                        doc_type="receipt",
                        language="vi",
                        timeout=10,
                    )

            self.assertEqual(samples_out.read_text(encoding="utf-8"), "existing\n")
            self.assertFalse((root / ".samples.jsonl.tmp").exists())
