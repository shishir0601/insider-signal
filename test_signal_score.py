import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signal_score import SCORE_COMPONENTS, compute_signal_score, explain_signal_score


class TestComputeSignalScore(unittest.TestCase):
    def test_all_zero_features_score_zero(self):
        features = pd.DataFrame([{
            "dollar_volume_zscore": 0, "distinct_buyers_5d": 0,
            "role_weighted_buy_value_5d": 0, "purchases_per_buyer_5d": 0, "buy_sell_ratio_5d": 0,
        }])
        self.assertEqual(compute_signal_score(features).iloc[0], 0.0)

    def test_maxed_out_features_score_100(self):
        features = pd.DataFrame([{
            "dollar_volume_zscore": 100, "distinct_buyers_5d": 100,
            "role_weighted_buy_value_5d": 100_000_000, "purchases_per_buyer_5d": 100, "buy_sell_ratio_5d": 100,
        }])
        self.assertEqual(compute_signal_score(features).iloc[0], 100.0)

    def test_score_never_exceeds_100_or_goes_negative(self):
        features = pd.DataFrame([
            {"dollar_volume_zscore": 1e9, "distinct_buyers_5d": 1e9, "role_weighted_buy_value_5d": 1e12,
             "purchases_per_buyer_5d": 1e9, "buy_sell_ratio_5d": 1e9},
            {"dollar_volume_zscore": -50, "distinct_buyers_5d": -50, "role_weighted_buy_value_5d": -50,
             "purchases_per_buyer_5d": -50, "buy_sell_ratio_5d": -50},
        ])
        scores = compute_signal_score(features)
        self.assertTrue((scores <= 100).all())
        self.assertTrue((scores >= 0).all())

    def test_component_caps_sum_to_100(self):
        self.assertEqual(sum(cap for _, cap, _ in SCORE_COMPONENTS), 100)

    def test_missing_columns_contribute_zero_not_an_error(self):
        # A feature table missing some columns (e.g. hand-built test
        # data) should still produce a score, just a lower one.
        features = pd.DataFrame([{"dollar_volume_zscore": 5.0}])
        score = compute_signal_score(features)
        self.assertEqual(score.iloc[0], 35.0)  # only the zscore component contributes

    def test_more_abnormal_zscore_never_decreases_the_score(self):
        low = pd.DataFrame([{"dollar_volume_zscore": 1.0}])
        high = pd.DataFrame([{"dollar_volume_zscore": 4.0}])
        self.assertGreaterEqual(compute_signal_score(high).iloc[0], compute_signal_score(low).iloc[0])

    def test_nan_values_treated_as_zero_contribution(self):
        features = pd.DataFrame([{"dollar_volume_zscore": float("nan"), "distinct_buyers_5d": 6}])
        score = compute_signal_score(features)
        self.assertEqual(score.iloc[0], 25.0)  # only distinct_buyers_5d contributes


class TestExplainSignalScore(unittest.TestCase):
    def test_breakdown_sums_to_the_computed_score(self):
        row = pd.Series({
            "dollar_volume_zscore": 3.0, "distinct_buyers_5d": 4,
            "role_weighted_buy_value_5d": 500_000, "purchases_per_buyer_5d": 1.5, "buy_sell_ratio_5d": 0.8,
        })
        total_score = compute_signal_score(pd.DataFrame([row])).iloc[0]
        breakdown_total = round(sum(points for _, points, _ in explain_signal_score(row)), 1)
        self.assertAlmostEqual(breakdown_total, total_score, places=1)

    def test_breakdown_has_one_entry_per_component(self):
        row = pd.Series({"dollar_volume_zscore": 1.0})
        self.assertEqual(len(explain_signal_score(row)), len(SCORE_COMPONENTS))

    def test_missing_field_in_row_contributes_zero_points(self):
        row = pd.Series({"dollar_volume_zscore": 5.0})  # everything else missing
        breakdown = dict((label, points) for label, points, _ in explain_signal_score(row))
        self.assertEqual(breakdown["Number of independent insiders buying"], 0.0)


if __name__ == "__main__":
    unittest.main()
