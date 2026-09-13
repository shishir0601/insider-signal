"""
client.py

Talks to SEC EDGAR: resolves a ticker to a CIK, lists a company's Form 4
filings, and fetches one filing's XML. Plain `requests` — no scraping
framework, no XBRL library — because this project only needs these
three narrow calls.

SEC's fair-access policy requires every request to declare who's making
it (https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
— unidentified or generic-default traffic gets a 403 and can trigger a
temporary IP block. This client refuses to guess a default: construct
EdgarClient with a `user_agent` string of the form
"YourName your@email.com", or set the EDGAR_USER_AGENT environment
variable. Shipping a plausible-looking placeholder default is exactly
what gets everyone who forgets to change it blocked together.

Rate limit: SEC allows roughly 10 requests/second per IP. This client
waits `min_request_interval_seconds` (default 0.2s = 5/sec) between
calls — comfortably under that, with room for the caller's own retries.
"""

import os
import time
from typing import Optional

import requests

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}"

FORM4_FORM_TYPES = {"4", "4/A"}


class EdgarClientError(RuntimeError):
    """Raised for anything the caller should see plainly: a missing
    User-Agent, an HTTP error, or a ticker EDGAR doesn't recognize.
    Never swallowed into an empty result — an empty result and "this
    ticker doesn't exist" are different things a caller needs to tell apart."""


class EdgarClient:
    def __init__(self, user_agent: Optional[str] = None, min_request_interval_seconds: float = 0.2,
                 session: Optional[requests.Session] = None):
        user_agent = user_agent or os.environ.get("EDGAR_USER_AGENT")
        if not user_agent:
            raise EdgarClientError(
                "EdgarClient requires a User-Agent identifying you to SEC EDGAR, e.g. "
                "'YourName your@email.com' — pass user_agent=... or set the "
                "EDGAR_USER_AGENT environment variable. SEC blocks requests without "
                "one; see the README for why this isn't defaulted for you."
            )
        self._user_agent = user_agent
        self._min_interval = min_request_interval_seconds
        self._session = session or requests.Session()
        self._last_request_at = 0.0

    def _get(self, url: str) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        try:
            resp = self._session.get(url, headers={"User-Agent": self._user_agent}, timeout=30)
        finally:
            self._last_request_at = time.monotonic()
        if resp.status_code == 403:
            raise EdgarClientError(
                f"SEC EDGAR returned 403 Forbidden for {url} — usually a missing/rejected "
                "User-Agent, or a temporary block from too many requests. Wait a few "
                "minutes and confirm your User-Agent looks like 'Name email@domain'."
            )
        resp.raise_for_status()
        return resp

    def resolve_cik(self, ticker: str) -> str:
        """Ticker -> zero-padded 10-digit CIK string, via SEC's bulk
        ticker file (one request; callers doing many lookups should
        cache this response themselves rather than re-fetching per ticker)."""
        data = self._get(TICKER_LOOKUP_URL).json()
        ticker_upper = ticker.upper()
        for row in data.values():
            if row.get("ticker", "").upper() == ticker_upper:
                return f"{row['cik_str']:010d}"
        raise EdgarClientError(f"no CIK found for ticker {ticker!r}")

    def list_form4_filings(self, cik: str, limit: Optional[int] = None) -> list:
        """
        Returns a list of dicts — {accession_number, filing_date,
        report_date, form, primary_document} — for this CIK's Form 4 and
        4/A filings, most recent first.

        Limitation: only reads `filings.recent`. A CIK with a long filing
        history has older filings paginated into separate files this
        client doesn't follow — documented in the README, not hidden.
        """
        cik_padded = f"{int(cik):010d}"
        data = self._get(SUBMISSIONS_URL.format(cik=cik_padded)).json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])

        results = []
        for i, form in enumerate(forms):
            if form not in FORM4_FORM_TYPES:
                continue
            results.append({
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent.get("reportDate", [None] * len(forms))[i],
                "form": form,
                "primary_document": recent["primaryDocument"][i],
            })
            if limit and len(results) >= limit:
                break
        return results

    def fetch_filing_xml(self, cik: str, accession_number: str, primary_document: str) -> str:
        """Fetches one filing's primary XML document as text."""
        cik_nolead = str(int(cik))
        accession_nodash = accession_number.replace("-", "")
        url = ARCHIVE_URL.format(cik=cik_nolead, accession_nodash=accession_nodash, primary_doc=primary_document)
        return self._get(url).text
