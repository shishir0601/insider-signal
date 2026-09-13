import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.compare import DETECTORS, compare_detectors
from src.features import build_features
from src.generate_synthetic_data import generate


class TestCompareDetectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One real, moderately-sized synthetic run shared across these
        # tests -- expensive enough (walk-forward IF) that re-running it
        # per test would make the suite noticeably slower for no benefit.
        cls.tx, cls.prices, cls.ground_truth = generate(n_tickers=10, n_days=150, n_informed_clusters=3, seed=3)
        cls.features = build_features(cls.tx, prices=cls.prices)
        cls.results = compare_detectors(cls.features, cls.prices, cls.ground_truth, seed=3)

    def test_returns_one_row_per_detector(self):
        self.assertEqual(set(self.results.index), set(DETECTORS.keys()))

    def test_all_expected_columns_present(self):
        for col in ["n_flagged", "flagged_pct", "mean_return_flagged", "p_value",
                    "precision", "recall", "f1", "pr_auc", "roc_auc"]:
            self.assertIn(col, self.results.columns)

    def test_random_baseline_is_not_statistically_significant_or_precise(self):
        # A sanity check on the framework itself: random flagging should
        # not look good on either evaluation lens (if it does, something
        # in the comparison plumbing is broken, not the random detector).
        random_row = self.results.loc["Random baseline"]
        if random_row["p_value"] is not None:
            self.assertGreater(random_row["p_value"], 0.01)
        if random_row["roc_auc"] is not None:
            self.assertLess(random_row["roc_auc"], 0.65)

    def test_real_detectors_beat_random_on_ground_truth_recall_or_precision(self):
        random_row = self.results.loc["Random baseline"]
        for label in ["Statistical baseline", "Isolation Forest (walk-forward)"]:
            row = self.results.loc[label]
            if row["precision"] is not None and random_row["precision"] is not None:
                self.assertGreaterEqual(row["precision"], random_row["precision"])

    def test_statistical_baseline_flag_count_matches_direct_call(self):
        from src.detect import statistical_baseline
        direct = statistical_baseline(self.features)
        self.assertEqual(self.results.loc["Statistical baseline", "n_flagged"], int(direct.is_anomaly.sum()))

    def test_handles_a_run_with_zero_injected_clusters_gracefully(self):
        tx, prices, ground_truth = generate(n_tickers=5, n_days=90, n_informed_clusters=0, seed=9)
        features = build_features(tx, prices=prices)
        results = compare_detectors(features, prices, ground_truth, seed=9)
        # No ground truth positives -> precision/recall are None, not a crash.
        self.assertTrue(results["precision"].isna().all() or results["precision"].isnull().all())


if __name__ == "__main__":
    unittest.main()
