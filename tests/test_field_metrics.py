from __future__ import annotations

import unittest

from eval.field_metrics import document_fields_exact, field_value_accuracy


class FieldMetricTests(unittest.TestCase):
    def test_field_value_accuracy_is_per_field_macro_average(self) -> None:
        result = field_value_accuracy({"total": "10.00", "date": "wrong"}, {"total": "10.00", "date": "2026-01-01"})
        self.assertEqual(result, 0.5)

    def test_document_fields_exact_requires_all_required_fields(self) -> None:
        self.assertEqual(document_fields_exact({"total": "10.00"}, {"total": "10.00", "date": "2026-01-01"}), 0.0)
        self.assertEqual(document_fields_exact({"total": "10.00", "date": "2026-01-01"}, {"total": "10.00", "date": "2026-01-01"}), 1.0)
