"""
Tests for src/cleaning.py — one test per rule, plus the guarantee that
synthetic data (already clean by construction) passes through unchanged.
"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cleaning import clean_transactions
from src.generate_synthetic_data import generate

GOOD_ROW = {"ticker": "AAPL", "insider_id": "X1", "date": "2024-01-05",
            "transaction_type": "BUY", "shares": 100, "price": 10.0}


def _df(*rows):
    return pd.DataFrame(rows)


class TestCleaning(unittest.TestCase):
    def test_synthetic_data_passes_through_unchanged(self):
        tx, _, _ = generate(seed=42)
        cleaned, report = clean_transactions(tx)
        self.assertEqual(len(cleaned), len(tx))
        self.assertEqual(report.total_dropped, 0)

    def test_drops_row_missing_a_required_field(self):
        dirty = _df(GOOD_ROW, {**GOOD_ROW, "insider_id": None, "ticker": "MSFT"})
        cleaned, report = clean_transactions(dirty)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.dropped_by_reason["missing a required field"], 1)

    def test_drops_non_numeric_shares(self):
        dirty = _df(GOOD_ROW, {**GOOD_ROW, "ticker": "MSFT", "shares": "not-a-number"})
        cleaned, report = clean_transactions(dirty)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.dropped_by_reason["non-numeric or non-positive shares/price"], 1)

    def test_drops_negative_price(self):
        dirty = _df(GOOD_ROW, {**GOOD_ROW, "ticker": "MSFT", "price": -5.0})
        cleaned, report = clean_transactions(dirty)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.dropped_by_reason["non-numeric or non-positive shares/price"], 1)

    def test_drops_zero_shares(self):
        dirty = _df(GOOD_ROW, {**GOOD_ROW, "ticker": "MSFT", "shares": 0})
        cleaned, _ = clean_transactions(dirty)
        self.assertEqual(len(cleaned), 1)

    def test_drops_unparseable_date(self):
        dirty = _df(GOOD_ROW, {**GOOD_ROW, "ticker": "MSFT", "date": "not-a-date"})
        cleaned, report = clean_transactions(dirty)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.dropped_by_reason["invalid or implausible transaction date"], 1)

    def test_drops_implausibly_old_date(self):
        dirty = _df(GOOD_ROW, {**GOOD_ROW, "ticker": "MSFT", "date": "1776-07-04"})
        cleaned, report = clean_transactions(dirty)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.dropped_by_reason["invalid or implausible transaction date"], 1)

    def test_does_not_reject_dates_just_because_they_are_after_todays_wall_clock(self):
        # Regression guard: cleaning must not depend on when the code
        # happens to run. A date far in the future is still implausible,
        # but a date that is merely "after today" is not, by itself, a
        # data quality problem (e.g. backfilled/synthetic data, or a
        # deliberately future-dated test dataset).
        far_future_but_plausible = "2090-01-01"
        dirty = _df({**GOOD_ROW, "date": far_future_but_plausible})
        cleaned, report = clean_transactions(dirty)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.total_dropped, 0)

    def test_drops_unrecognized_transaction_type(self):
        dirty = _df(GOOD_ROW, {**GOOD_ROW, "ticker": "MSFT", "transaction_type": "GRANT"})
        cleaned, report = clean_transactions(dirty)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.dropped_by_reason["transaction_type not BUY/SELL"], 1)

    def test_normalizes_ticker_case_and_whitespace(self):
        dirty = _df({**GOOD_ROW, "ticker": " aapl "})
        cleaned, _ = clean_transactions(dirty)
        self.assertEqual(cleaned.iloc[0].ticker, "AAPL")

    def test_duplicate_rows_after_normalization_are_dropped(self):
        # These two rows are identical once ticker casing is normalized —
        # must be caught as one duplicate, not missed because of formatting.
        dirty = _df(GOOD_ROW, {**GOOD_ROW, "ticker": " aapl "})
        cleaned, report = clean_transactions(dirty)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.dropped_by_reason["exact duplicate transaction"], 1)

    def test_distinct_transactions_are_not_treated_as_duplicates(self):
        distinct = _df(GOOD_ROW, {**GOOD_ROW, "shares": 200})
        cleaned, report = clean_transactions(distinct)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report.total_dropped, 0)

    def test_min_purchase_value_filter_is_disabled_by_default(self):
        tiny = _df({**GOOD_ROW, "shares": 1, "price": 0.01})
        cleaned, report = clean_transactions(tiny)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report.total_dropped, 0)

    def test_min_purchase_value_filter_when_enabled(self):
        tiny = _df({**GOOD_ROW, "shares": 1, "price": 0.01}, {**GOOD_ROW, "ticker": "MSFT"})
        cleaned, report = clean_transactions(tiny, min_purchase_value=100)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0].ticker, "MSFT")
        self.assertEqual(list(report.dropped_by_reason.values()), [1])

    def test_empty_input_does_not_crash(self):
        empty = pd.DataFrame(columns=["ticker", "insider_id", "date", "transaction_type", "shares", "price"])
        cleaned, report = clean_transactions(empty)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(report.input_rows, 0)

    def test_report_summary_is_readable_text(self):
        dirty = _df(GOOD_ROW, {**GOOD_ROW, "ticker": "MSFT", "shares": -1})
        _, report = clean_transactions(dirty)
        summary = report.summary()
        self.assertIn("1 rows out", summary)
        self.assertIn("non-numeric or non-positive shares/price", summary)


if __name__ == "__main__":
    unittest.main()
