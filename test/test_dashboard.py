"""
Tests for the Phase 4 dashboard. Uses Flask's test client (no live
server needed) to exercise the exact request/response path a browser
would hit. Kept intentionally small, matching the dashboard itself —
this checks the main flow and the edge cases that would otherwise show
up as a broken page, not every possible input.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard import create_app
from dashboard.data_service import get_demo_dataset


class DashboardTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.demo = get_demo_dataset()

    def test_homepage_loads_with_search_form(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Analyze", resp.data)

    def test_demo_ticker_with_signals_renders_full_page(self):
        # TCK00 is known (from prior manual runs) to have a detected
        # cluster at the default seed.
        resp = self.client.get("/?ticker=TCK00")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("TCK00", html)
        self.assertIn("Demo data", html)
        self.assertIn("standard deviations above", html)  # the explanation sentence
        self.assertIn("Signal score breakdown", html)
        self.assertIn("data:image/png;base64", html)
        self.assertNotIn("Traceback", html)
        self.assertNotIn("{{", html)  # no unrendered Jinja

    def test_ticker_input_is_case_insensitive(self):
        resp = self.client.get("/?ticker=tck00")
        self.assertIn("TCK00", resp.data.decode())

    def test_ticker_with_no_flagged_signals_has_no_broken_images(self):
        # Find a demo ticker with zero flagged rows at the default seed.
        features = self.demo["features"]
        no_signal_tickers = [
            t for t in self.demo["tickers"]
            if not features[(features.ticker == t) & features.flagged_any].shape[0]
        ]
        if not no_signal_tickers:
            self.skipTest("every demo ticker has at least one flagged signal at this seed")
        resp = self.client.get(f"/?ticker={no_signal_tickers[0]}")
        html = resp.data.decode()
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn('src=""', html)
        self.assertIn("No flagged days", html)
        self.assertNotIn("Signal detail", html)  # no signal to show detail for

    def test_unknown_ticker_shows_not_found_and_demo_suggestions(self):
        resp = self.client.get("/?ticker=ZZZZ")
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode()
        self.assertIn("was not found", html)
        self.assertIn("TCK00", html)  # a demo ticker suggestion link

    def test_empty_ticker_param_shows_landing_state_not_an_error(self):
        resp = self.client.get("/?ticker=")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("Traceback", resp.data.decode())

    def test_all_five_chart_types_present_for_a_ticker_with_signals(self):
        resp = self.client.get("/?ticker=TCK00")
        html = resp.data.decode()
        chart_imgs = re.findall(r'<img src="(data:image/png;base64,[^"]+)"', html)
        # price+signals, forward returns, history, cumulative, comparison
        self.assertEqual(len(chart_imgs), 5)

    def test_explanation_numbers_match_the_underlying_feature_row(self):
        # The explanation sentence must be built from real data, not a
        # canned string -- spot-check the insider count actually
        # mentioned matches what data_service computed.
        from dashboard.data_service import analyze_ticker
        result = analyze_ticker("TCK00")
        self.assertIsNotNone(result["top_signal"])
        n = result["top_signal"]["distinct_buyers"]
        self.assertIn(f"{n} distinct insider", result["top_signal"]["explanation"])


if __name__ == "__main__":
    unittest.main()
