"""
Tests for the Phase 2 features added to src/features.py. Phase 1's
test_pipeline.py already covers the four original features
(distinct_buyers_5d, buy_dollar_volume_5d, buy_sell_ratio_5d,
dollar_volume_zscore) and is left untouched.
"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import build_features, _role_weight

BASE_TX = {"ticker": "X", "transaction_type": "BUY", "shares": 100, "price": 10.0}


def _tx(**overrides):
    row = dict(BASE_TX)
    row.update(overrides)
    return row


class TestPurchaseCountAndConcentration(unittest.TestCase):
    def test_n_purchases_counts_trades_not_distinct_insiders(self):
        # 2 insiders, but insider A trades twice -> 3 purchases, 2 distinct buyers.
        tx = pd.DataFrame([
            _tx(insider_id="A", date="2024-01-05", insider_title=""),
            _tx(insider_id="A", date="2024-01-08", insider_title=""),
            _tx(insider_id="B", date="2024-01-08", insider_title=""),
        ])
        feat = build_features(tx)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2024-01-08"))].iloc[0]
        self.assertEqual(row.n_purchases_5d, 3)
        self.assertEqual(row.distinct_buyers_5d, 2)

    def test_purchases_per_buyer_is_the_ratio(self):
        tx = pd.DataFrame([
            _tx(insider_id="A", date="2024-01-05", insider_title=""),
            _tx(insider_id="A", date="2024-01-08", insider_title=""),
            _tx(insider_id="B", date="2024-01-08", insider_title=""),
        ])
        feat = build_features(tx)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2024-01-08"))].iloc[0]
        self.assertAlmostEqual(row.purchases_per_buyer_5d, 3 / 2)

    def test_purchases_per_buyer_is_zero_with_no_buyers(self):
        tx = pd.DataFrame([_tx(insider_id="A", date="2024-01-05", transaction_type="SELL", insider_title="")])
        feat = build_features(tx)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2024-01-05"))].iloc[0]
        self.assertEqual(row.purchases_per_buyer_5d, 0.0)


class TestRoleWeighting(unittest.TestCase):
    def test_role_weight_matches_known_titles(self):
        self.assertEqual(_role_weight("Chief Executive Officer", {"chief executive": 3.0}), 3.0)
        self.assertEqual(_role_weight("Director", {"director": 2.0}), 2.0)

    def test_role_weight_defaults_to_one_for_unknown_or_missing_title(self):
        self.assertEqual(_role_weight("Assistant Regional Manager", {"chief executive": 3.0}), 1.0)
        self.assertEqual(_role_weight("", {"chief executive": 3.0}), 1.0)
        self.assertEqual(_role_weight(None, {"chief executive": 3.0}), 1.0)

    def test_ceo_purchase_weighted_higher_than_unweighted_dollar_volume(self):
        tx = pd.DataFrame([_tx(insider_id="A", date="2024-01-05", insider_title="Chief Executive Officer")])
        feat = build_features(tx, role_weights={"chief executive": 3.0})
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2024-01-05"))].iloc[0]
        self.assertAlmostEqual(row.role_weighted_buy_value_5d, 1000 * 3.0)  # 100 shares * $10 * weight 3

    def test_missing_insider_title_column_falls_back_to_unweighted(self):
        # No insider_title column at all (e.g. older callers/hand-built data)
        # should not crash, and should behave as if every weight were 1.0.
        tx = pd.DataFrame([{"ticker": "X", "insider_id": "A", "date": "2024-01-05",
                             "transaction_type": "BUY", "shares": 100, "price": 10.0}])
        feat = build_features(tx)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2024-01-05"))].iloc[0]
        self.assertAlmostEqual(row.role_weighted_buy_value_5d, row.buy_dollar_volume_5d)


class TestInsiderRelativeSizeZscore(unittest.TestCase):
    def test_zero_with_insufficient_insider_history(self):
        # Only 2 prior trades for insider A; default min is 3 -> stays 0.
        tx = pd.DataFrame([
            _tx(insider_id="A", date="2024-01-02", shares=100, insider_title=""),
            _tx(insider_id="A", date="2024-01-03", shares=100, insider_title=""),
            _tx(insider_id="A", date="2024-01-04", shares=100_000, insider_title=""),  # huge, but not enough history yet
        ])
        feat = build_features(tx, insider_history_min_trades=3)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2024-01-04"))].iloc[0]
        self.assertEqual(row.insider_relative_size_zscore_5d, 0.0)

    def test_large_trade_relative_to_own_history_scores_positive(self):
        # Insider A makes several small, consistent trades, then one much
        # bigger one -- that last trade should score a clearly positive
        # z-score relative to THEIR OWN history once enough history exists.
        rows = [_tx(insider_id="A", date=f"2024-01-0{i}", shares=100, insider_title="") for i in range(1, 6)]
        rows.append(_tx(insider_id="A", date="2024-01-08", shares=100_000, insider_title=""))
        tx = pd.DataFrame(rows)
        feat = build_features(tx, insider_history_min_trades=3)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2024-01-08"))].iloc[0]
        self.assertGreater(row.insider_relative_size_zscore_5d, 0)

    def test_does_not_compare_against_a_different_insiders_history(self):
        # Insider B's one small trade should not be judged unusual just
        # because insider A trades in a totally different size range.
        tx = pd.DataFrame([
            _tx(insider_id="A", date="2024-01-02", shares=100_000, insider_title=""),
            _tx(insider_id="A", date="2024-01-03", shares=100_000, insider_title=""),
            _tx(insider_id="A", date="2024-01-04", shares=100_000, insider_title=""),
            _tx(insider_id="B", date="2024-01-05", shares=100, insider_title=""),
        ])
        feat = build_features(tx, insider_history_min_trades=3)
        row = feat[(feat.ticker == "X") & (feat.date == pd.Timestamp("2024-01-05"))].iloc[0]
        # B has no prior history at all -> 0.0, not judged against A.
        self.assertEqual(row.insider_relative_size_zscore_5d, 0.0)


class TestPreSignalReturn(unittest.TestCase):
    def test_zero_when_no_prices_given(self):
        tx = pd.DataFrame([_tx(insider_id="A", date="2024-01-05", insider_title="")])
        feat = build_features(tx)  # no prices=
        row = feat.iloc[0]
        self.assertEqual(row.pre_signal_return_10d, 0.0)

    def test_computes_backward_looking_return_when_prices_given(self):
        dates = pd.bdate_range("2024-01-01", periods=15)
        prices = pd.DataFrame({"ticker": "X", "date": dates, "close": [100.0] * 10 + [110.0] * 5})
        tx = pd.DataFrame([_tx(insider_id="A", date=str(dates[12].date()), insider_title="")])
        feat = build_features(tx, prices=prices, pre_signal_return_window_days=10)
        row = feat[feat.date == dates[12]].iloc[0]
        # price 10 trading days before dates[12] is 100, price at dates[12] is 110
        self.assertAlmostEqual(row.pre_signal_return_10d, 0.10, places=4)


class TestNoLeakage(unittest.TestCase):
    """Every feature for day d must depend only on transactions/prices
    dated on or before d. These tests perturb data strictly AFTER a
    fixed observation day and confirm that day's features don't change."""

    def _base_data(self):
        insider_dates = pd.bdate_range("2024-02-01", periods=3)
        rows = [
            _tx(insider_id=insider, date=str(d.date()), shares=500, insider_title="Director")
            for insider, d in zip(["A", "B", "C"], insider_dates)
        ]
        tx = pd.DataFrame(rows)
        dates = pd.bdate_range("2024-01-01", periods=40)
        prices = pd.DataFrame({"ticker": "X", "date": dates, "close": 100.0 + pd.Series(range(40)) * 0.1})
        return tx, prices, insider_dates[-1]

    def test_future_transactions_do_not_change_past_feature_row(self):
        tx, prices, observation_day = self._base_data()

        feat_before = build_features(tx, prices=prices)
        row_before = feat_before[feat_before.date == observation_day].iloc[0]

        future_tx = pd.concat([tx, pd.DataFrame([
            _tx(insider_id="Z", date="2024-02-20", shares=999_999, insider_title="Chief Executive Officer"),
        ])], ignore_index=True)
        feat_after = build_features(future_tx, prices=prices)
        row_after = feat_after[feat_after.date == observation_day].iloc[0]

        for col in ["distinct_buyers_5d", "buy_dollar_volume_5d", "buy_sell_ratio_5d",
                    "dollar_volume_zscore", "n_purchases_5d", "purchases_per_buyer_5d",
                    "insider_relative_size_zscore_5d", "role_weighted_buy_value_5d"]:
            self.assertEqual(row_before[col], row_after[col], f"{col} changed when a FUTURE transaction was added")

    def test_future_prices_do_not_change_past_pre_signal_return(self):
        tx, prices, observation_day = self._base_data()

        feat_before = build_features(tx, prices=prices)
        row_before = feat_before[feat_before.date == observation_day].iloc[0]

        altered_prices = prices.copy()
        altered_prices.loc[altered_prices["date"] > observation_day, "close"] *= 5  # blow up future prices
        feat_after = build_features(tx, prices=altered_prices)
        row_after = feat_after[feat_after.date == observation_day].iloc[0]

        self.assertAlmostEqual(row_before.pre_signal_return_10d, row_after.pre_signal_return_10d)

    def test_a_later_insiders_huge_trade_does_not_leak_into_an_earlier_row(self):
        # Specifically targets insider_relative_size_zscore_5d: an
        # insider's OWN future trade must not retroactively change how
        # their earlier trade is scored.
        tx = pd.DataFrame([
            _tx(insider_id="A", date="2024-01-02", shares=100, insider_title=""),
            _tx(insider_id="A", date="2024-01-03", shares=100, insider_title=""),
            _tx(insider_id="A", date="2024-01-04", shares=100, insider_title=""),
            _tx(insider_id="A", date="2024-01-05", shares=150, insider_title=""),
        ])
        feat_before = build_features(tx, insider_history_min_trades=3)
        row_before = feat_before[feat_before.date == pd.Timestamp("2024-01-05")].iloc[0]

        tx_with_future_spike = pd.concat([tx, pd.DataFrame([
            _tx(insider_id="A", date="2024-01-10", shares=500_000, insider_title=""),
        ])], ignore_index=True)
        feat_after = build_features(tx_with_future_spike, insider_history_min_trades=3)
        row_after = feat_after[feat_after.date == pd.Timestamp("2024-01-05")].iloc[0]

        self.assertEqual(row_before.insider_relative_size_zscore_5d, row_after.insider_relative_size_zscore_5d)


if __name__ == "__main__":
    unittest.main()
