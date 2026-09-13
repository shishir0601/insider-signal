"""
ingest.py

Ties the EDGAR client, XML parser, and normalizer together into the one
function most callers actually want: "give me this ticker's insider
transactions, in the same schema the rest of the pipeline uses."

    from src.edgar.client import EdgarClient
    from src.edgar.ingest import fetch_transactions_for_ticker

    client = EdgarClient(user_agent="YourName your@email.com")
    transactions = fetch_transactions_for_ticker(client, "AAPL", limit=50)

This is ingestion only — it does not clean the result (see
src/cleaning.py) or run it through features.py. Each step stays a
separate, explicit call so it can be tested and reasoned about on its
own, per the project's "ingestion -> cleaning -> normalized schema ->
features -> detection" architecture.
"""

import logging
from typing import Optional

import pandas as pd

from src.edgar.client import EdgarClient
from src.edgar.normalize import normalize_edgar_records
from src.edgar.parser import parse_form4_xml

logger = logging.getLogger(__name__)


def fetch_transactions_for_ticker(client: EdgarClient, ticker: str, limit: Optional[int] = 50) -> pd.DataFrame:
    """
    limit caps how many of the ticker's most recent Form 4 filings are
    fetched (each filing is its own HTTP request) — default 50 keeps a
    single call to this function well-behaved; raise it deliberately for
    a full historical pull.
    """
    cik = client.resolve_cik(ticker)
    filings = client.list_form4_filings(cik, limit=limit)

    all_records = []
    for filing in filings:
        try:
            xml_text = client.fetch_filing_xml(cik, filing["accession_number"], filing["primary_document"])
            all_records.extend(parse_form4_xml(xml_text))
        except Exception:
            # One bad filing (malformed XML, an unexpected variant)
            # shouldn't take down a pull of dozens of filings — log it
            # and continue. This is the only place in the ingestion path
            # that swallows an exception, and it's logged, not silent.
            logger.warning(
                "Skipping unparseable Form 4 filing %s for %s",
                filing["accession_number"], ticker, exc_info=True,
            )

    return normalize_edgar_records(all_records)
