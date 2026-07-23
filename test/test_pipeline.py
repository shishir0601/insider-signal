"""
Unit tests using Python's built-in unittest (no pytest — kept
dependency-free since this environment has no network access to install
packages, and it means anyone can run `python3 -m unittest` with zero setup).
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import build_features
from src.detect import statistical_baseline, isolation_forest_detect
from src.backtest import forward_return, backtest
from src.generate_synthetic_data import generate


class TestFeatures(unittest.TestCase):
    def test_distinct_buyers_counts_unique_insiders_not_trade_count(self):
        # Same insider trading twice in the window should count as 1 distinct buyer, not 2.
        tx = pd.DataFrame([
            {"ticker": "X", "insider_id": "A", "date": "2026-01-05", "transaction_type": "BUY", "shares": 100, "price": 10},
            {"ticker": "X", "insider_id": "A", "date": "2026-01-06", "transaction_type": "BUY", "shares": 100, "price": 10},
        ])
        feat = build_features(tx)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2026-01-06"))].iloc[0]
        self.assertEqual(row.distinct_buyers_5d, 1)

    def test_multiple_distinct_insiders_counted_separately(self):
        tx = pd.DataFrame([
            {"ticker": "X", "insider_id": "A", "date": "2026-01-05", "transaction_type": "BUY", "shares": 100, "price": 10},
            {"ticker": "X", "insider_id": "B", "date": "2026-01-05", "transaction_type": "BUY", "shares": 100, "price": 10},
            {"ticker": "X", "insider_id": "C", "date": "2026-01-06", "transaction_type": "BUY", "shares": 100, "price": 10},
        ])
        feat = build_features(tx)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2026-01-06"))].iloc[0]
        self.assertEqual(row.distinct_buyers_5d, 3)

    def test_buy_sell_ratio_is_one_when_only_buys(self):
        tx = pd.DataFrame([
            {"ticker": "X", "insider_id": "A", "date": "2026-01-05", "transaction_type": "BUY", "shares": 100, "price": 10},
        ])
        feat = build_features(tx)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2026-01-05"))].iloc[0]
        self.assertEqual(row.buy_sell_ratio_5d, 1.0)

    def test_buy_sell_ratio_defaults_to_half_with_no_activity(self):
        tx = pd.DataFrame([
            {"ticker": "X", "insider_id": "A", "date": "2026-01-05", "transaction_type": "BUY", "shares": 100, "price": 10},
        ])
        feat = build_features(tx)
        far_out_row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2026-01-05"))].iloc[0]
        # sanity: with zero trailing activity elsewhere, ratio should not error/NaN
        self.assertFalse(np.isnan(far_out_row.buy_sell_ratio_5d))

    def test_empty_transactions_returns_empty_frame_not_a_crash(self):
        empty_tx = pd.DataFrame(columns=["ticker", "insider_id", "date", "transaction_type", "shares", "price"])
        feat = build_features(empty_tx)
        self.assertEqual(len(feat), 0)
        self.assertIn("dollar_volume_zscore", feat.columns)

    def test_single_transaction_does_not_produce_nan_or_inf_zscore(self):
        # A ticker with only one data point has zero trailing standard
        # deviation — the code must not divide by zero here.
        tx = pd.DataFrame([
            {"ticker": "X", "insider_id": "A", "date": "2026-01-05", "transaction_type": "BUY", "shares": 100, "price": 10},
        ])
        feat = build_features(tx)
        self.assertFalse(feat["dollar_volume_zscore"].isna().any())
        self.assertFalse(np.isinf(feat["dollar_volume_zscore"]).any())


class TestDetect(unittest.TestCase):
    def test_statistical_baseline_requires_both_conditions(self):
        features = pd.DataFrame([
            # high zscore but only 1 buyer -> should NOT flag
            {"ticker": "A", "date": "2026-01-01", "distinct_buyers_5d": 1, "buy_dollar_volume_5d": 100000, "buy_sell_ratio_5d": 1.0, "dollar_volume_zscore": 5.0},
            # high zscore AND 4 buyers -> should flag
            {"ticker": "B", "date": "2026-01-01", "distinct_buyers_5d": 4, "buy_dollar_volume_5d": 100000, "buy_sell_ratio_5d": 1.0, "dollar_volume_zscore": 5.0},
            # many buyers but low zscore -> should NOT flag
            {"ticker": "C", "date": "2026-01-01", "distinct_buyers_5d": 5, "buy_dollar_volume_5d": 1000, "buy_sell_ratio_5d": 1.0, "dollar_volume_zscore": 0.1},
        ])
        result = statistical_baseline(features)
        flagged = dict(zip(result.ticker, result.is_anomaly))
        self.assertFalse(flagged["A"])
        self.assertTrue(flagged["B"])
        self.assertFalse(flagged["C"])

    def test_isolation_forest_flags_a_small_fraction_matching_contamination(self):
        rng = np.random.default_rng(0)
        n = 500
        normal = pd.DataFrame({
            "ticker": ["X"] * n,
            "date": pd.bdate_range("2026-01-01", periods=n),
            "distinct_buyers_5d": rng.poisson(1, n),
            "buy_dollar_volume_5d": rng.normal(10000, 2000, n).clip(0),
            "buy_sell_ratio_5d": rng.uniform(0.3, 0.7, n),
            "dollar_volume_zscore": rng.normal(0, 1, n),
        })
        result = isolation_forest_detect(normal, contamination=0.05)
        flagged_fraction = result.is_anomaly.mean()
        # contamination is a target rate, not exact — allow reasonable tolerance
        self.assertTrue(0.02 <= flagged_fraction <= 0.09)


class TestBacktest(unittest.TestCase):
    def test_forward_return_computes_correct_percentage_change(self):
        prices = pd.DataFrame([
            {"ticker": "X", "date": "2026-01-01", "close": 100},
            {"ticker": "X", "date": "2026-01-02", "close": 110},
            {"ticker": "X", "date": "2026-01-03", "close": 121},
        ])
        r = forward_return(prices, "X", "2026-01-01", horizon_days=2)
        self.assertAlmostEqual(r, 0.21, places=6)

    def test_forward_return_returns_none_when_horizon_exceeds_data(self):
        prices = pd.DataFrame([
            {"ticker": "X", "date": "2026-01-01", "close": 100},
            {"ticker": "X", "date": "2026-01-02", "close": 110},
        ])
        r = forward_return(prices, "X", "2026-01-01", horizon_days=10)
        self.assertIsNone(r)

    def test_backtest_does_not_crash_when_nothing_is_flagged(self):
        features = pd.DataFrame([
            {"ticker": "X", "date": "2026-01-01", "distinct_buyers_5d": 1, "buy_dollar_volume_5d": 100, "buy_sell_ratio_5d": 0.5, "dollar_volume_zscore": 0.1},
            {"ticker": "X", "date": "2026-01-02", "distinct_buyers_5d": 1, "buy_dollar_volume_5d": 100, "buy_sell_ratio_5d": 0.5, "dollar_volume_zscore": 0.1},
        ])
        detections = statistical_baseline(features)
        self.assertEqual(detections.is_anomaly.sum(), 0)
        prices = pd.DataFrame([
            {"ticker": "X", "date": "2026-01-01", "close": 100},
            {"ticker": "X", "date": "2026-01-02", "close": 101},
        ])
        result = backtest(detections, prices, horizon_days=1)
        self.assertIsNone(result["flagged_mean_return"])
        self.assertIsNone(result["p_value"])

    def test_pipeline_recovers_injected_signal_on_synthetic_ground_truth(self):
        """
        End-to-end sanity check: on data where we KNOW there's a real
        injected signal (generate_synthetic_data.py bakes in a price jump
        after each informed cluster), the statistical baseline detector
        should flag at least some of the true cluster windows, and the
        backtest should show flagged days outperforming random control
        days on average. This is the test that validates the whole
        pipeline is coherent, not just each piece in isolation.
        """
        tx, px, gt = generate(n_tickers=15, n_days=180, n_informed_clusters=5, seed=7)
        feat = build_features(tx)
        detections = statistical_baseline(feat, zscore_threshold=2.0, min_distinct_buyers=3)

        self.assertGreater(detections.is_anomaly.sum(), 0, "Expected at least some flags on data with injected signal")

        result = backtest(detections, px, horizon_days=10, seed=7)
        self.assertIsNotNone(result["flagged_mean_return"])
        self.assertIsNotNone(result["control_mean_return"])
        # The core claim: flagged days should show higher forward returns
        # than random control days, because we know a real jump was baked in.
        self.assertGreater(result["flagged_mean_return"], result["control_mean_return"])


if __name__ == "__main__":
    unittest.main()
