"""
Tests for the EDGAR ingestion layer. Parsing/normalization tests run
against the two real, saved Form 4 fixtures in test/fixtures/ (one
clean open-market purchase filing, one with an option exercise, tax
withholding, and a non-transaction holding) — no live network calls,
so these are fast and deterministic. Client tests mock `requests` so
they don't depend on SEC's servers being reachable or unchanged.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.edgar.client import EdgarClient, EdgarClientError
from src.edgar.normalize import normalize_edgar_records
from src.edgar.parser import parse_form4_xml

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return f.read()


class TestParser(unittest.TestCase):
    def test_parses_open_market_purchase_filing(self):
        rows = parse_form4_xml(_load_fixture("form4_open_market_purchase.xml"))
        self.assertEqual(len(rows), 2)
        row = rows[0]
        self.assertEqual(row["issuer_ticker"], "AMR")
        self.assertEqual(row["issuer_name"], "Alpha Metallurgical Resources, Inc.")
        self.assertEqual(row["insider_cik"], "0001812746")
        self.assertEqual(row["insider_name"], "Courtis Kenneth S.")
        self.assertEqual(row["insider_title"], "Director")  # from isDirector, no officerTitle
        self.assertEqual(row["transaction_code"], "P")
        self.assertEqual(row["acquired_disposed_code"], "A")
        self.assertEqual(row["shares"], "390")
        self.assertEqual(row["price_per_share"], "175.48")

    def test_parses_officer_title_when_present(self):
        rows = parse_form4_xml(_load_fixture("form4_option_exercise.xml"))
        self.assertEqual(rows[0]["insider_title"], "President")

    def test_option_exercise_has_no_price_but_still_parses(self):
        # Real filings often omit the price for an "M" (exercise) line —
        # the parser should return None for it, not crash.
        rows = parse_form4_xml(_load_fixture("form4_option_exercise.xml"))
        exercise_row = next(r for r in rows if r["transaction_code"] == "M")
        self.assertIsNone(exercise_row["price_per_share"])
        self.assertEqual(exercise_row["shares"], "651")

    def test_ignores_nonderivative_holdings_and_derivative_transactions(self):
        # The Moderna fixture has 2 nonDerivativeTransaction, 1
        # nonDerivativeHolding, and 1 derivativeTransaction. Only the
        # two actual non-derivative transactions should come back.
        rows = parse_form4_xml(_load_fixture("form4_option_exercise.xml"))
        self.assertEqual(len(rows), 2)

    def test_raises_on_malformed_xml(self):
        with self.assertRaises(ValueError):
            parse_form4_xml("<not><valid</xml>")

    def test_raises_on_wrong_root_element(self):
        with self.assertRaises(ValueError):
            parse_form4_xml("<somethingElse></somethingElse>")

    def test_filing_with_no_transactions_returns_empty_list(self):
        xml = """<?xml version="1.0"?>
        <ownershipDocument>
            <issuer><issuerCik>1</issuerCik><issuerName>X</issuerName><issuerTradingSymbol>X</issuerTradingSymbol></issuer>
            <reportingOwner><reportingOwnerId><rptOwnerCik>2</rptOwnerCik><rptOwnerName>Y</rptOwnerName></reportingOwnerId></reportingOwner>
        </ownershipDocument>"""
        self.assertEqual(parse_form4_xml(xml), [])


class TestNormalize(unittest.TestCase):
    def _records(self):
        records = []
        for fname in ("form4_open_market_purchase.xml", "form4_option_exercise.xml"):
            records.extend(parse_form4_xml(_load_fixture(fname)))
        return records

    def test_produces_the_shared_internal_schema_columns(self):
        df = normalize_edgar_records(self._records())
        for col in ("ticker", "insider_id", "date", "transaction_type", "shares", "price"):
            self.assertIn(col, df.columns)

    def test_open_market_purchase_classified_as_buy_and_open_market(self):
        df = normalize_edgar_records(self._records())
        row = df[df.transaction_code == "P"].iloc[0]
        self.assertEqual(row.transaction_type, "BUY")
        self.assertTrue(row.is_open_market)

    def test_option_exercise_classified_as_buy_but_not_open_market(self):
        df = normalize_edgar_records(self._records())
        row = df[df.transaction_code == "M"].iloc[0]
        self.assertEqual(row.transaction_type, "BUY")  # acquired_disposed_code == A
        self.assertFalse(row.is_open_market)
        self.assertEqual(row.transaction_code_label, "Option exercise / RSU vesting")

    def test_tax_withholding_classified_as_sell_but_not_open_market(self):
        df = normalize_edgar_records(self._records())
        row = df[df.transaction_code == "F"].iloc[0]
        self.assertEqual(row.transaction_type, "SELL")
        self.assertFalse(row.is_open_market)

    def test_unrecognized_transaction_code_gets_a_fallback_label(self):
        df = normalize_edgar_records([{
            "issuer_ticker": "X", "insider_cik": "1", "transaction_date": "2024-01-01",
            "acquired_disposed_code": "A", "transaction_code": "Q", "shares": "1", "price_per_share": "1",
        }])
        self.assertEqual(df.iloc[0].transaction_code_label, "Other / unrecognized code")
        self.assertFalse(df.iloc[0].is_open_market)

    def test_empty_input_returns_empty_frame_with_correct_columns(self):
        df = normalize_edgar_records([])
        self.assertEqual(len(df), 0)
        self.assertIn("ticker", df.columns)


class TestEdgarClient(unittest.TestCase):
    def test_requires_a_user_agent(self):
        env = os.environ.copy()
        env.pop("EDGAR_USER_AGENT", None)
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(EdgarClientError):
                EdgarClient()

    def test_accepts_user_agent_from_environment_variable(self):
        with patch.dict(os.environ, {"EDGAR_USER_AGENT": "Test test@example.com"}):
            client = EdgarClient()
            self.assertEqual(client._user_agent, "Test test@example.com")

    def test_resolve_cik_matches_ticker_case_insensitively(self):
        mock_session = MagicMock()
        mock_session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}},
        )
        client = EdgarClient(user_agent="Test test@example.com", session=mock_session)
        self.assertEqual(client.resolve_cik("aapl"), "0000320193")

    def test_resolve_cik_raises_for_unknown_ticker(self):
        mock_session = MagicMock()
        mock_session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}},
        )
        client = EdgarClient(user_agent="Test test@example.com", session=mock_session)
        with self.assertRaises(EdgarClientError):
            client.resolve_cik("NOPE")

    def test_list_form4_filings_filters_to_form_4_and_4a(self):
        mock_session = MagicMock()
        mock_session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"filings": {"recent": {
                "form": ["4", "8-K", "4/A", "10-Q"],
                "accessionNumber": ["0001-24-000001", "0001-24-000002", "0001-24-000003", "0001-24-000004"],
                "filingDate": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "reportDate": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "primaryDocument": ["a.xml", "b.htm", "c.xml", "d.htm"],
            }}},
        )
        client = EdgarClient(user_agent="Test test@example.com", session=mock_session)
        filings = client.list_form4_filings("320193")
        self.assertEqual([f["form"] for f in filings], ["4", "4/A"])

    def test_get_raises_clear_error_on_403(self):
        mock_session = MagicMock()
        mock_session.get.return_value = MagicMock(status_code=403)
        client = EdgarClient(user_agent="Test test@example.com", session=mock_session)
        with self.assertRaises(EdgarClientError):
            client.resolve_cik("AAPL")


if __name__ == "__main__":
    unittest.main()
