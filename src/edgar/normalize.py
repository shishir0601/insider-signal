"""
normalize.py

Maps parser.py's raw per-transaction dicts into the same internal
schema src/generate_synthetic_data.py already produces:

    ticker, insider_id, date, transaction_type, shares, price

plus enrichment columns the synthetic generator has no equivalent for
(insider_name, insider_title, transaction_code, transaction_code_label,
is_open_market, total_value, issuer_name). build_features() and the
detectors only read the six columns above, so real and synthetic
transactions are interchangeable inputs to the rest of the pipeline —
the extra columns are additive, never required.

Classification is driven by two Form 4 fields together, not the
transaction code alone:
  - transactionAcquiredDisposedCode (A/D) gives the $ direction — this
    alone is enough for transaction_type, which is all
    build_features() needs.
  - transactionCode distinguishes *why* shares moved. P/S are open-
    market purchases/sales — the informative signal this project looks
    for. Other codes (M exercise, A award/grant, F tax withholding, G
    gift, ...) mean the shares moved for a reason that has little to do
    with the insider's private view of the stock. is_open_market flags
    this distinction; no detector in this phase uses it yet (feature
    engineering changes are out of scope for Phase 1 — see README).

Does not drop or fix anything questionable here (a missing price, a
non-numeric share count) — that's src/cleaning.py's job, run as a
separate, documented step, so nothing is ever silently excluded inside
a "normalize" call the caller didn't expect to filter anything.
"""

import pandas as pd

OPEN_MARKET_CODES = {"P", "S"}

TRANSACTION_CODE_LABELS = {
    "P": "Open market purchase",
    "S": "Open market sale",
    "A": "Grant/award",
    "M": "Option exercise / RSU vesting",
    "F": "Tax withholding",
    "G": "Gift",
    "C": "Conversion of derivative",
    "D": "Disposition to issuer",
    "I": "Discretionary transaction",
    "J": "Other acquisition/disposition",
    "K": "Equity swap",
    "L": "Small acquisition (Rule 16a-6)",
    "U": "Tender of shares",
    "W": "Acquisition/disposition by will or laws of descent",
    "Z": "Deposit/withdrawal from voting trust",
}

NORMALIZED_COLUMNS = [
    "ticker", "insider_id", "date", "transaction_type", "shares", "price",
    "insider_name", "insider_title", "transaction_code", "transaction_code_label",
    "is_open_market", "total_value", "issuer_name",
]

ACQUIRED_DISPOSED_TO_TYPE = {"A": "BUY", "D": "SELL"}


def normalize_edgar_records(raw_records: list) -> pd.DataFrame:
    """
    raw_records: a list of dicts from parser.parse_form4_xml() — typically
    the concatenated results of parsing many filings before calling this
    once (see src/edgar/ingest.py).
    """
    if not raw_records:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    df = pd.DataFrame(raw_records)

    transaction_type = df.get("acquired_disposed_code", pd.Series(dtype=object)).map(ACQUIRED_DISPOSED_TO_TYPE)
    transaction_code = df.get("transaction_code", pd.Series(dtype=object))
    is_open_market = transaction_code.isin(OPEN_MARKET_CODES)
    code_label = transaction_code.map(TRANSACTION_CODE_LABELS).fillna("Other / unrecognized code")

    shares = pd.to_numeric(df.get("shares"), errors="coerce")
    price = pd.to_numeric(df.get("price_per_share"), errors="coerce")

    return pd.DataFrame({
        "ticker": df.get("issuer_ticker"),
        "insider_id": df.get("insider_cik"),
        "date": df.get("transaction_date"),
        "transaction_type": transaction_type,
        "shares": shares,
        "price": price,
        "insider_name": df.get("insider_name"),
        "insider_title": df.get("insider_title"),
        "transaction_code": transaction_code,
        "transaction_code_label": code_label,
        "is_open_market": is_open_market,
        "total_value": shares * price,
        "issuer_name": df.get("issuer_name"),
    })
