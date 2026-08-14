from __future__ import annotations

import unittest

from eval.dataset_adapters import adapt_fields, adapt_sample, text_metrics_applicable


class DatasetAdapterTests(unittest.TestCase):
    def test_invoices_donut_flattens_scalar_header_and_summary_only(self) -> None:
        fields = {
            "header": {"invoice_no": "FV/1/2020", "invoice_date": "2020-01-01"},
            "items": [{"item_desc": "Service", "item_net_worth": "10.00"}],
            "summary": {"total_net_worth": "10.00", "total_gross_worth": "12.30"},
        }

        self.assertEqual(
            adapt_fields("invoices_donut", fields),
            {
                "header.invoice_no": "FV/1/2020",
                "header.invoice_date": "2020-01-01",
                "summary.total_net_worth": "10.00",
                "summary.total_gross_worth": "12.30",
            },
        )

    def test_structured_target_datasets_do_not_use_text_metrics(self) -> None:
        self.assertFalse(text_metrics_applicable("invoices_donut"))
        self.assertFalse(text_metrics_applicable("mychen76_invoices"))
        self.assertFalse(text_metrics_applicable("fake_w2"))
        self.assertTrue(text_metrics_applicable("sroie_2019"))

    def test_adapt_sample_keeps_metadata_and_replaces_fields(self) -> None:
        sample = {
            "sample_id": "invoice-1",
            "dataset_id": "invoices_donut",
            "gt_text": "flattened structured target",
            "gt_fields": {"header": {"invoice_no": "1"}, "items": [{"item_desc": "A"}]},
        }

        adapted = adapt_sample("invoices_donut", sample)

        self.assertEqual(adapted["sample_id"], "invoice-1")
        self.assertEqual(adapted["gt_text"], "flattened structured target")
        self.assertEqual(adapted["gt_fields"], {"header.invoice_no": "1"})
