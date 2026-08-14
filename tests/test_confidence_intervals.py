from __future__ import annotations

import unittest

from eval.confidence_intervals import bootstrap_mean_ci, quantile


class ConfidenceIntervalTests(unittest.TestCase):
    def test_type_7_quantile_interpolates(self) -> None:
        self.assertEqual(quantile([1.0, 3.0], 0.5), 2.0)

    def test_bootstrap_is_reproducible(self) -> None:
        first = bootstrap_mean_ci([0.0, 0.5, 1.0], resamples=100, seed=9)
        second = bootstrap_mean_ci([0.0, 0.5, 1.0], resamples=100, seed=9)
        self.assertEqual(first, second)

    def test_one_document_interval_is_degenerate(self) -> None:
        self.assertEqual(bootstrap_mean_ci([0.25]), (0.25, 0.25))
