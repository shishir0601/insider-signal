"""
cleaning.py

Validates a transactions DataFrame — from either
src/generate_synthetic_data.py or src/edgar/ingest.py, since both
produce the same schema — before it reaches features.py.

Every row this removes is counted and the reason recorded in the
returned CleaningReport; nothing is silently discarded. Synthetic data
is clean by construction, so running it through here is a no-op (see
test/test_cleaning.py) — this module exists for real data's sake, but
runs identically on either source because they share a schema.
"""

from dataclasses import dataclass, field

import pandas as pd

from src.config import DEFAULT_CONFIG

REQUIRED_COLUMNS = ["ticker", "insider_id", "date", "transaction_type", "shares", "price"]
VALID_TRANSACTION_TYPES = {"BUY", "SELL"}
DUPLICATE_KEY_COLUMNS = ["ticker", "insider_id", "date", "transaction_type", "shares", "price"]

# A sane bound for a transaction date, not a "must not be in the future"
# check. EDGAR electronic filing didn't exist before 1994, and a date
# that far out either way is far more likely to be a parsing error
# (a garbage string, a swapped field) than a real transaction. This is
# deliberately NOT tied to the wall-clock "now" — a batch job cleaning
# historical or backfilled data (or this project's own synthetic test
# fixtures, which use an arbitrary baseline year) shouldn't have rows
# start disappearing just because real-world time passed a fixed date.
EARLIEST_PLAUSIBLE_DATE = pd.Timestamp("1994-01-01")
LATEST_PLAUSIBLE_DATE = pd.Timestamp("2100-01-01")


@dataclass
class CleaningReport:
    input_rows: int = 0
    output_rows: int = 0
    dropped_by_reason: dict = field(default_factory=dict)

    def record_drop(self, reason: str, count: int) -> None:
        if count:
            self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + count

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped_by_reason.values())

    def summary(self) -> str:
        lines = [f"{self.input_rows} rows in, {self.output_rows} rows out "
                 f"({self.total_dropped} dropped)"]
        for reason, count in sorted(self.dropped_by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {count}: {reason}")
        return "\n".join(lines)


def clean_transactions(transactions: pd.DataFrame,
                        min_purchase_value: float = DEFAULT_CONFIG.min_purchase_value):
    """
    Returns (cleaned_df, CleaningReport).

    Cleaning steps, in this order (order matters — a row missing a
    required field can't be checked for validity, so it's removed
    first; ticker/insider_id text is normalized before duplicate
    detection so "AAPL" and " aapl " are recognized as the same entity
    instead of hiding a real duplicate):

      1. missing a required field                     -> dropped
      2. shares/price not numeric, zero, or negative   -> dropped
      3. date doesn't parse, or is implausible          -> dropped
      4. transaction_type isn't BUY or SELL            -> dropped
      5. ticker/insider_id whitespace + case           -> normalized, not dropped
      6. below min_purchase_value (if configured > 0)  -> dropped
      7. exact duplicate of an earlier row              -> dropped

    Does NOT reconcile Form 4/A amendments to the specific transaction
    they correct (that needs accession-level filing lineage this
    schema doesn't carry) — an amendment that restates rather than
    exactly repeats a transaction will pass through as a second, distinct
    row. Documented as a known limitation, not silently handled as if solved.
    """
    report = CleaningReport(input_rows=len(transactions))
    df = transactions.copy()

    # 1. required fields present.
    missing_mask = df[REQUIRED_COLUMNS].isna().any(axis=1)
    report.record_drop("missing a required field", int(missing_mask.sum()))
    df = df[~missing_mask]

    # 2. shares/price numeric and positive.
    shares_numeric = pd.to_numeric(df["shares"], errors="coerce")
    price_numeric = pd.to_numeric(df["price"], errors="coerce")
    invalid_numeric_mask = (
        shares_numeric.isna() | price_numeric.isna() | (shares_numeric <= 0) | (price_numeric <= 0)
    )
    report.record_drop("non-numeric or non-positive shares/price", int(invalid_numeric_mask.sum()))
    df = df[~invalid_numeric_mask]
    shares_numeric = shares_numeric[~invalid_numeric_mask]
    price_numeric = price_numeric[~invalid_numeric_mask]

    # 3. dates must parse and fall in a plausible range. Deliberately not
    # "must not be after today" — see EARLIEST/LATEST_PLAUSIBLE_DATE above.
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid_date_mask = (
        parsed_dates.isna() | (parsed_dates < EARLIEST_PLAUSIBLE_DATE) | (parsed_dates > LATEST_PLAUSIBLE_DATE)
    )
    report.record_drop("invalid or implausible transaction date", int(invalid_date_mask.sum()))
    df = df[~invalid_date_mask]
    shares_numeric = shares_numeric[~invalid_date_mask]
    price_numeric = price_numeric[~invalid_date_mask]
    parsed_dates = parsed_dates[~invalid_date_mask]

    df = df.assign(shares=shares_numeric, price=price_numeric, date=parsed_dates)

    # 4. transaction_type must be something build_features() understands.
    valid_type_mask = df["transaction_type"].isin(VALID_TRANSACTION_TYPES)
    report.record_drop("transaction_type not BUY/SELL", int((~valid_type_mask).sum()))
    df = df[valid_type_mask]

    # 5. normalize ticker/insider_id formatting so the same entity in two
    # filings (or two data sources) is recognized as one.
    df = df.assign(
        ticker=df["ticker"].astype(str).str.strip().str.upper(),
        insider_id=df["insider_id"].astype(str).str.strip(),
    )

    # 6. optional minimum purchase value (see config.py) — real Form 4
    # data includes trivial transactions that aren't meaningful signal.
    # Disabled by default (0) and not needed for synthetic data.
    if min_purchase_value > 0:
        value = df["shares"] * df["price"]
        below_min_mask = value < min_purchase_value
        report.record_drop(f"below minimum purchase value (${min_purchase_value:,.0f})", int(below_min_mask.sum()))
        df = df[~below_min_mask]

    # 7. exact duplicate transactions (e.g. the same filing ingested
    # twice). Keeps the first occurrence.
    dup_mask = df.duplicated(subset=DUPLICATE_KEY_COLUMNS, keep="first")
    report.record_drop("exact duplicate transaction", int(dup_mask.sum()))
    df = df[~dup_mask]

    report.output_rows = len(df)
    return df.reset_index(drop=True), report
