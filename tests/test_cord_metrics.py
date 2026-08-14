from __future__ import annotations

import unittest

from eval.cord_metrics import (
    cord_document_exact,
    cord_menu_column_scores,
    cord_menu_line_item_scores,
    cord_scalar_fields,
    normalize_cord_value,
)


class CordMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ground_truth = {
            "menu": [
                {"nm": "Coffee", "cnt": "1 x", "price": "60.000"},
                {"nm": "Coffee", "cnt": "1 x", "price": "60.000"},
            ],
            "sub_total": {"subtotal_price": "120.000", "tax_price": "5.455"},
            "total": {"total_price": "125.455", "menuqty_cnt": "2.00"},
        }

    def test_scalar_adapter_flattens_and_normalizes_amounts(self) -> None:
        predicted = {
            "sub_total": {"subtotal_price": "120,000", "tax_price": "5,455"},
            "total.total_price": "125,455",
            "total.menuqty_cnt": "2",
        }
        self.assertEqual(cord_scalar_fields(predicted), cord_scalar_fields(self.ground_truth))
        self.assertEqual(normalize_cord_value("total.total_price", "60.000"), "60000")
        self.assertEqual(normalize_cord_value("total.total_price", "60,000"), "60000")
        self.assertNotEqual(normalize_cord_value("total.total_price", "12.30"), "1230")

    def test_menu_scores_are_order_independent_and_duplicate_aware(self) -> None:
        predicted = {
            "menu": [
                {"nm": "COFFEE", "cnt": "1", "price": "60,000"},
                {"nm": "Coffee", "cnt": "1 x", "price": "60.000"},
            ]
        }
        scores = cord_menu_line_item_scores(predicted, self.ground_truth)
        self.assertEqual(scores["precision"], 1.0)
        self.assertEqual(scores["recall"], 1.0)
        self.assertEqual(scores["f1"], 1.0)

    def test_menu_column_scores_diagnose_partial_line_item_matches(self) -> None:
        predicted = {
            "menu": [
                {"nm": "COFFEE", "cnt": "1", "price": "60,000"},
                {"nm": "Coffee", "cnt": "2", "price": "99,000"},
            ]
        }
        scores = cord_menu_column_scores(predicted, self.ground_truth)
        self.assertEqual(scores["nm"]["f1"], 1.0)
        self.assertEqual(scores["cnt"]["recall"], 0.5)
        self.assertEqual(scores["cnt"]["precision"], 0.5)
        self.assertEqual(scores["price"]["recall"], 0.5)

    def test_document_exact_requires_scalars_and_all_menu_rows(self) -> None:
        predicted = {
            "menu": [
                {"nm": "Coffee", "cnt": "1", "price": "60,000"},
                {"nm": "Coffee", "cnt": "1", "price": "60,000"},
            ],
            "sub_total": {"subtotal_price": "120,000", "tax_price": "5,455"},
            "total": {"total_price": "125,455", "menuqty_cnt": "2"},
        }
        self.assertEqual(cord_document_exact(predicted, self.ground_truth), 1.0)
        predicted["total"]["total_price"] = "125,000"
        self.assertEqual(cord_document_exact(predicted, self.ground_truth), 0.0)
