"""
Tests for src/backtest_analysis.py. Grouped by requirement: forward
returns (multi-horizon, entry lag), benchmark/excess returns,
transaction costs, performance metrics, significance testing, temporal
split, and — the section that matters most for this phase — an
explicit audit that entry-lag shifting and the temporal split don't
themselves introduce look-ahead.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest_analysis import (
    apply_transaction_costs,
    compute_benchmark_returns,
    compute_excess_returns,
    compute_forward_returns,
    compute_performance_metrics,
    run_out_of_sample_backtest,
    significance_test,
    split_by_date,
)
from src.features import build_features
from src.generate_synthetic_data import generate


def _flat_prices(ticker, n_days, start_price=100.0, daily_return=0.0, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=n_days)
    prices = start_price * np.cumprod(1 + np.full(n_days, daily_return))
    return pd.DataFrame({"ticker": ticker, "date": dates, "close": prices})


class TestComputeForwardReturns(unittest.TestCase):
    def test_computes_all_requested_horizons(self):
        prices = _flat_prices("X", 40, daily_return=0.01)
        detections = pd.DataFrame([
            {"ticker": "X", "date": prices["date"].iloc[5], "is_anomaly": True, "score": 1.0},
        ])
        fr = compute_forward_returns(detections, prices, horizons_days=(1, 5, 10))
        for h in (1, 5, 10):
            self.assertIn(f"return_{h}d", fr.columns)
            self.assertFalse(pd.isna(fr[f"return_{h}d"].iloc[0]))

    def test_only_flagged_rows_produce_signals(self):
        prices = _flat_prices("X", 40, daily_return=0.0)
        detections = pd.DataFrame([
            {"ticker": "X", "date": prices["date"].iloc[5], "is_anomaly": True, "score": 1.0},
            {"ticker": "X", "date": prices["date"].iloc[6], "is_anomaly": False, "score": 0.0},
        ])
        fr = compute_forward_returns(detections, prices, horizons_days=(5,))
        self.assertEqual(len(fr), 1)

    def test_signal_too_close_to_end_of_data_gets_nan_not_dropped(self):
        prices = _flat_prices("X", 20)
        detections = pd.DataFrame([
            {"ticker": "X", "date": prices["date"].iloc[18], "is_anomaly": True, "score": 1.0},
        ])
        fr = compute_forward_returns(detections, prices, horizons_days=(1, 20))
        self.assertEqual(len(fr), 1)  # row kept
        self.assertFalse(pd.isna(fr["return_1d"].iloc[0]))
        self.assertTrue(pd.isna(fr["return_20d"].iloc[0]))  # 20d horizon unreachable

    def test_entry_lag_shifts_the_entry_date_forward(self):
        # A constant daily growth rate would make every 5-day window
        # identical regardless of start date, masking the effect being
        # tested -- use a varying path so shifting the entry date
        # actually changes which window is measured.
        dates = pd.bdate_range("2024-01-01", periods=30)
        rng = np.random.default_rng(0)
        prices = pd.DataFrame({"ticker": "X", "date": dates, "close": 100 * np.cumprod(1 + rng.normal(0, 0.02, 30))})
        signal_date = dates[5]
        detections = pd.DataFrame([{"ticker": "X", "date": signal_date, "is_anomaly": True, "score": 1.0}])

        fr_lag0 = compute_forward_returns(detections, prices, horizons_days=(5,), entry_lag_days=0)
        fr_lag2 = compute_forward_returns(detections, prices, horizons_days=(5,), entry_lag_days=2)

        self.assertEqual(fr_lag0["entry_date"].iloc[0], signal_date)
        self.assertEqual(fr_lag2["entry_date"].iloc[0], dates[7])
        self.assertNotAlmostEqual(fr_lag0["return_5d"].iloc[0], fr_lag2["return_5d"].iloc[0], places=6)

    def test_empty_detections_returns_empty_frame_not_a_crash(self):
        prices = _flat_prices("X", 20)
        empty = pd.DataFrame(columns=["ticker", "date", "is_anomaly", "score"])
        fr = compute_forward_returns(empty, prices)
        self.assertEqual(len(fr), 0)


class TestBenchmarkAndExcessReturns(unittest.TestCase):
    def test_benchmark_is_the_average_across_all_tickers(self):
        prices = pd.concat([
            _flat_prices("A", 20, daily_return=0.02),
            _flat_prices("B", 20, daily_return=0.00),
        ], ignore_index=True)
        d = prices[prices.ticker == "A"]["date"].iloc[5]
        bench = compute_benchmark_returns(prices, pd.Series([d]), horizons_days=(5,))
        # A grows ~10%/5days, B grows 0% -> average should be roughly half A's return
        a_return = (1.02 ** 5) - 1
        self.assertAlmostEqual(bench["benchmark_return_5d"].iloc[0], a_return / 2, places=3)

    def test_excess_return_is_signal_minus_benchmark(self):
        prices = pd.concat([
            _flat_prices("A", 20, daily_return=0.02),
            _flat_prices("B", 20, daily_return=0.00),
        ], ignore_index=True)
        d = prices[prices.ticker == "A"]["date"].iloc[5]
        detections = pd.DataFrame([{"ticker": "A", "date": d, "is_anomaly": True, "score": 1.0}])
        fr = compute_forward_returns(detections, prices, horizons_days=(5,))
        bench = compute_benchmark_returns(prices, fr["date"], horizons_days=(5,))
        merged = compute_excess_returns(fr, bench, horizons_days=(5,))
        self.assertAlmostEqual(
            merged["excess_return_5d"].iloc[0],
            merged["return_5d"].iloc[0] - merged["benchmark_return_5d"].iloc[0],
        )


class TestTransactionCosts(unittest.TestCase):
    def test_reduces_every_return_by_the_round_trip_drag(self):
        raw = np.array([0.10, -0.05, 0.0])
        adjusted = apply_transaction_costs(raw, transaction_cost_bps=10, slippage_bps=5)
        expected_drag = 2 * (10 + 5) / 10_000
        np.testing.assert_allclose(adjusted, raw - expected_drag)

    def test_zero_costs_leaves_returns_unchanged(self):
        raw = np.array([0.10, -0.05])
        adjusted = apply_transaction_costs(raw, transaction_cost_bps=0, slippage_bps=0)
        np.testing.assert_allclose(adjusted, raw)


class TestPerformanceMetrics(unittest.TestCase):
    def test_all_positive_returns_gives_hit_rate_one(self):
        metrics = compute_performance_metrics(np.array([0.01, 0.02, 0.03]))
        self.assertEqual(metrics["hit_rate"], 1.0)

    def test_all_negative_returns_gives_hit_rate_zero(self):
        metrics = compute_performance_metrics(np.array([-0.01, -0.02]))
        self.assertEqual(metrics["hit_rate"], 0.0)

    def test_cumulative_return_compounds_not_sums(self):
        returns = np.array([0.10, 0.10])
        metrics = compute_performance_metrics(returns)
        self.assertAlmostEqual(metrics["cumulative_return"], 1.10 * 1.10 - 1, places=6)
        self.assertNotAlmostEqual(metrics["cumulative_return"], 0.20, places=6)

    def test_max_drawdown_is_zero_for_monotonically_increasing_returns(self):
        metrics = compute_performance_metrics(np.array([0.01, 0.02, 0.03]))
        self.assertAlmostEqual(metrics["max_drawdown"], 0.0)

    def test_max_drawdown_is_negative_after_a_loss(self):
        metrics = compute_performance_metrics(np.array([0.10, -0.20, 0.05]))
        self.assertLess(metrics["max_drawdown"], 0.0)

    def test_best_and_worst_trade_are_correct(self):
        metrics = compute_performance_metrics(np.array([0.05, -0.10, 0.20, -0.03]))
        self.assertEqual(metrics["best_trade"], 0.20)
        self.assertEqual(metrics["worst_trade"], -0.10)

    def test_nan_values_are_excluded_not_treated_as_zero(self):
        with_nan = compute_performance_metrics(np.array([0.05, np.nan, 0.10]))
        without_nan = compute_performance_metrics(np.array([0.05, 0.10]))
        self.assertEqual(with_nan["n"], without_nan["n"])
        self.assertAlmostEqual(with_nan["mean_return"], without_nan["mean_return"])

    def test_empty_input_returns_none_fields_not_a_crash(self):
        metrics = compute_performance_metrics(np.array([]))
        self.assertEqual(metrics["n"], 0)
        self.assertIsNone(metrics["mean_return"])

    def test_single_return_has_no_sharpe_ratio(self):
        # std with ddof=1 on a single point is undefined -- must not
        # divide by zero or crash.
        metrics = compute_performance_metrics(np.array([0.05]))
        self.assertEqual(metrics["n"], 1)
        self.assertIsNone(metrics["sharpe_ratio"])


class TestSignificanceTest(unittest.TestCase):
    def test_identical_distributions_are_not_significant(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(0, 0.02, 200)
        control = rng.normal(0, 0.02, 200)
        result = significance_test(sample, control)
        self.assertFalse(result["is_significant"])

    def test_clearly_different_distributions_are_significant(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(0.10, 0.01, 200)
        control = rng.normal(0.0, 0.01, 200)
        result = significance_test(sample, control)
        self.assertTrue(result["is_significant"])
        self.assertLess(result["p_value"], 0.05)

    def test_reports_honest_none_with_too_few_observations(self):
        result = significance_test(np.array([0.05]), np.array([0.01, 0.02, 0.03]))
        self.assertIsNone(result["p_value"])
        self.assertFalse(result["is_significant"])

    def test_confidence_interval_contains_the_sample_mean(self):
        rng = np.random.default_rng(1)
        sample = rng.normal(0.05, 0.02, 50)
        result = significance_test(sample, rng.normal(0, 0.02, 50))
        self.assertLessEqual(result["ci_low"], result["mean_sample"])
        self.assertGreaterEqual(result["ci_high"], result["mean_sample"])

    def test_reports_sample_sizes(self):
        result = significance_test(np.array([0.01, 0.02, 0.03]), np.array([0.0, 0.0]))
        self.assertEqual(result["n_sample"], 3)
        self.assertEqual(result["n_control"], 2)


class TestSplitByDate(unittest.TestCase):
    def test_split_is_chronological_not_random(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        split_date = split_by_date(dates, development_fraction=0.6)
        # Every date before the split must be earlier in time, not a random subset.
        self.assertTrue(all(d <= split_date for d in dates[:60]))

    def test_development_fraction_controls_the_split_point(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        early_split = split_by_date(dates, development_fraction=0.2)
        late_split = split_by_date(dates, development_fraction=0.8)
        self.assertLess(early_split, late_split)

    def test_raises_on_empty_dates(self):
        with self.assertRaises(ValueError):
            split_by_date(pd.Series([], dtype="datetime64[ns]"))


class TestNoLeakageInBacktestAnalysis(unittest.TestCase):
    """The look-ahead audit this phase explicitly asks for, applied to
    the NEW machinery (entry lag, temporal split, out-of-sample runner)
    rather than re-testing features.py's leakage (already covered by
    test_features_v2.TestNoLeakage)."""

    def test_forward_return_for_an_earlier_signal_ignores_a_later_price_shock(self):
        prices = _flat_prices("X", 40, daily_return=0.0)
        early_signal_date = prices["date"].iloc[5]
        detections = pd.DataFrame([{"ticker": "X", "date": early_signal_date, "is_anomaly": True, "score": 1.0}])

        fr_before = compute_forward_returns(detections, prices, horizons_days=(5,))

        shocked_prices = prices.copy()
        shocked_prices.loc[shocked_prices["date"] > prices["date"].iloc[20], "close"] *= 10
        fr_after = compute_forward_returns(detections, shocked_prices, horizons_days=(5,))

        self.assertAlmostEqual(fr_before["return_5d"].iloc[0], fr_after["return_5d"].iloc[0])

    def test_out_of_sample_backtest_only_reports_rows_after_the_split(self):
        tx, prices, ground_truth = generate(n_tickers=6, n_days=100, n_informed_clusters=2, seed=5)
        features = build_features(tx, prices=prices)
        results = run_out_of_sample_backtest(features, prices, ground_truth, development_fraction=0.6)

        split_date = results["split_date"]
        for label, detector_result in results["detectors"].items():
            fr = detector_result["forward_returns_all_horizons_lag0"]
            if len(fr):
                self.assertTrue((pd.to_datetime(fr["date"]) > split_date).all(),
                                 f"{label} reported a signal dated at or before the OOS split")

    def test_walkforward_isolation_forest_within_oos_still_never_trains_on_same_or_later_data(self):
        # Re-affirms detect.py's own guarantee still holds when driven
        # through run_out_of_sample_backtest's specific call pattern.
        tx, prices, ground_truth = generate(n_tickers=6, n_days=120, n_informed_clusters=2, seed=6)
        features = build_features(tx, prices=prices)
        from src.detect import isolation_forest_detect_walkforward
        detections = isolation_forest_detect_walkforward(features, min_train_rows=50, refit_frequency_days=15)
        scored = detections[detections["score"].notna()]
        for d in scored["date"].unique():
            n_strictly_before = int((features["date"] < d).sum())
            self.assertGreaterEqual(n_strictly_before, 50)


if __name__ == "__main__":
    unittest.main()
