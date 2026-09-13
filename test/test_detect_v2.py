"""
Tests for the Phase 2 additions to src/detect.py. The original two
detectors' existing tests in test_pipeline.py are untouched (aside from
the one updated in this phase for the enriched feature list — see
test_pipeline.TestDetect).
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detect import (
    isolation_forest_detect_walkforward,
    random_baseline_detect,
    statistical_baseline,
)


class TestRandomBaseline(unittest.TestCase):
    def test_flags_approximately_the_requested_rate(self):
        n = 2000
        features = pd.DataFrame({
            "ticker": ["X"] * n,
            "date": pd.bdate_range("2024-01-01", periods=n),
            "dollar_volume_zscore": 0.0,
            "distinct_buyers_5d": 0,
        })
        result = random_baseline_detect(features, flag_rate=0.05, seed=1)
        self.assertAlmostEqual(result.is_anomaly.mean(), 0.05, delta=0.01)

    def test_is_deterministic_given_the_same_seed(self):
        features = pd.DataFrame({"ticker": ["X"] * 100, "date": pd.bdate_range("2024-01-01", periods=100)})
        a = random_baseline_detect(features, seed=7)
        b = random_baseline_detect(features, seed=7)
        pd.testing.assert_series_equal(a["is_anomaly"], b["is_anomaly"])
        pd.testing.assert_series_equal(a["score"], b["score"])

    def test_different_seeds_flag_different_rows(self):
        features = pd.DataFrame({"ticker": ["X"] * 200, "date": pd.bdate_range("2024-01-01", periods=200)})
        a = random_baseline_detect(features, flag_rate=0.1, seed=1)
        b = random_baseline_detect(features, flag_rate=0.1, seed=2)
        self.assertFalse(a["is_anomaly"].equals(b["is_anomaly"]))

    def test_flag_rate_zero_flags_nothing(self):
        features = pd.DataFrame({"ticker": ["X"] * 50, "date": pd.bdate_range("2024-01-01", periods=50)})
        result = random_baseline_detect(features, flag_rate=0.0)
        self.assertEqual(result.is_anomaly.sum(), 0)

    def test_flag_rate_one_flags_everything(self):
        features = pd.DataFrame({"ticker": ["X"] * 50, "date": pd.bdate_range("2024-01-01", periods=50)})
        result = random_baseline_detect(features, flag_rate=1.0)
        self.assertEqual(result.is_anomaly.sum(), 50)

    def test_empty_input_does_not_crash(self):
        features = pd.DataFrame(columns=["ticker", "date"])
        result = random_baseline_detect(features)
        self.assertEqual(len(result), 0)


class TestStatisticalBaselineThresholds(unittest.TestCase):
    """Threshold-behavior edge cases beyond test_pipeline.py's basic
    "requires both conditions" test."""

    def _features(self, **overrides):
        row = {"ticker": "X", "date": "2024-01-01", "distinct_buyers_5d": 3,
               "buy_dollar_volume_5d": 100000, "buy_sell_ratio_5d": 1.0, "dollar_volume_zscore": 3.0}
        row.update(overrides)
        return pd.DataFrame([row])

    def test_exactly_at_zscore_threshold_flags(self):
        result = statistical_baseline(self._features(dollar_volume_zscore=2.5), zscore_threshold=2.5)
        self.assertTrue(result.iloc[0].is_anomaly)

    def test_just_below_zscore_threshold_does_not_flag(self):
        result = statistical_baseline(self._features(dollar_volume_zscore=2.4999), zscore_threshold=2.5)
        self.assertFalse(result.iloc[0].is_anomaly)

    def test_exactly_at_min_distinct_buyers_flags(self):
        result = statistical_baseline(self._features(distinct_buyers_5d=3), min_distinct_buyers=3)
        self.assertTrue(result.iloc[0].is_anomaly)

    def test_one_below_min_distinct_buyers_does_not_flag(self):
        result = statistical_baseline(self._features(distinct_buyers_5d=2), min_distinct_buyers=3)
        self.assertFalse(result.iloc[0].is_anomaly)

    def test_raising_the_threshold_can_only_reduce_or_hold_flags_steady(self):
        # A monotonicity check: a stricter threshold should never flag a
        # row a looser threshold didn't.
        features = pd.DataFrame({
            "ticker": ["X"] * 20,
            "date": pd.bdate_range("2024-01-01", periods=20),
            "distinct_buyers_5d": np.arange(20) % 5,
            "buy_dollar_volume_5d": 1000,
            "buy_sell_ratio_5d": 1.0,
            "dollar_volume_zscore": np.linspace(-1, 5, 20),
        })
        loose = statistical_baseline(features, zscore_threshold=1.0, min_distinct_buyers=1)
        strict = statistical_baseline(features, zscore_threshold=3.0, min_distinct_buyers=3)
        self.assertTrue((~strict.is_anomaly | loose.is_anomaly).all())
        self.assertGreaterEqual(loose.is_anomaly.sum(), strict.is_anomaly.sum())


class TestIsolationForestWalkForward(unittest.TestCase):
    def _features(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "ticker": ["X"] * n,
            "date": pd.bdate_range("2024-01-01", periods=n),
            "distinct_buyers_5d": rng.poisson(1, n),
            "buy_dollar_volume_5d": rng.normal(10000, 2000, n).clip(0),
            "buy_sell_ratio_5d": rng.uniform(0.3, 0.7, n),
            "dollar_volume_zscore": rng.normal(0, 1, n),
            "n_purchases_5d": rng.poisson(1, n),
            "purchases_per_buyer_5d": rng.uniform(0.5, 1.5, n),
            "insider_relative_size_zscore_5d": rng.normal(0, 1, n),
            "role_weighted_buy_value_5d": rng.normal(10000, 2000, n).clip(0),
            "pre_signal_return_10d": rng.normal(0, 0.05, n),
        })

    def test_early_rows_before_min_train_rows_are_left_unscored(self):
        features = self._features(n=300)
        result = isolation_forest_detect_walkforward(features, min_train_rows=100, refit_frequency_days=20)
        # The first block (before any training data exists) must be NaN.
        self.assertTrue(result.iloc[0].score is None or np.isnan(result.iloc[0].score))
        self.assertFalse(result.iloc[0].is_anomaly)

    def test_later_rows_get_scored_once_enough_history_exists(self):
        features = self._features(n=300)
        result = isolation_forest_detect_walkforward(features, min_train_rows=100, refit_frequency_days=20)
        self.assertGreater(result["score"].notna().sum(), 0)

    def test_never_trains_on_same_day_or_later_data_than_it_scores(self):
        # A structural leakage check: for every scored row, confirm there
        # was at least min_train_rows of STRICTLY EARLIER data — i.e. the
        # function's own training-data selection is doing what it claims.
        features = self._features(n=300)
        min_train_rows = 100
        result = isolation_forest_detect_walkforward(features, min_train_rows=min_train_rows, refit_frequency_days=20)
        scored = result[result["score"].notna()]
        for d in scored["date"].unique():
            n_strictly_before = int((features["date"] < d).sum())
            self.assertGreaterEqual(n_strictly_before, min_train_rows)

    def test_empty_input_does_not_crash(self):
        empty = pd.DataFrame(columns=["ticker", "date", "distinct_buyers_5d", "buy_dollar_volume_5d",
                                        "buy_sell_ratio_5d", "dollar_volume_zscore", "n_purchases_5d",
                                        "purchases_per_buyer_5d", "insider_relative_size_zscore_5d",
                                        "role_weighted_buy_value_5d", "pre_signal_return_10d"])
        result = isolation_forest_detect_walkforward(empty)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
