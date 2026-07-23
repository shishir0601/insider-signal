"""
features.py

Turns raw Form 4 transactions into a per-(ticker, day) feature table that
an anomaly detector can actually use. The key insight: a single insider
buying isn't unusual — insiders buy their own stock all the time for
mundane reasons. What's unusual is CLUSTERING: several DIFFERENT insiders
at the same company buying independently within a tight window. That's
much harder to explain away as routine.

Features computed per (ticker, date), using a trailing window:
  - distinct_buyers_5d: how many different insiders bought in the past 5
    trading days (the core "cluster" signal)
  - buy_dollar_volume_5d: total $ value of insider buys in that window
  - buy_sell_ratio_5d: buy $ / (buy $ + sell $) — skews toward 1 when
    insiders are unusually one-sided
  - dollar_volume_zscore: buy_dollar_volume_5d expressed as a z-score
    against THAT TICKER's own trailing 60-day history — this is what
    lets us compare a normally-quiet stock to a normally-active one on
    the same footing, instead of flagging big companies just for being big.
"""

import numpy as np
import pandas as pd

TRAILING_WINDOW_DAYS = 5
BASELINE_WINDOW_DAYS = 60


FEATURE_COLUMNS = ["ticker", "date", "distinct_buyers_5d", "buy_dollar_volume_5d", "buy_sell_ratio_5d", "dollar_volume_zscore"]


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        # No transactions at all — return an empty, correctly-shaped table
        # rather than crashing on pd.bdate_range(NaT, NaT). A real pipeline
        # can hit this on a ticker with no filings in the lookback window.
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    transactions = transactions.copy()
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["dollar_value"] = transactions["shares"] * transactions["price"]

    all_dates = pd.bdate_range(transactions["date"].min(), transactions["date"].max())
    tickers = transactions["ticker"].unique()

    rows = []
    for ticker in tickers:
        tx = transactions[transactions["ticker"] == ticker]
        buys = tx[tx.transaction_type == "BUY"]

        # Daily $ aggregates via simple reindexed groupby — cheap and clear.
        daily = tx.groupby("date").agg(
            buy_dollar=("dollar_value", lambda s: s[tx.loc[s.index, "transaction_type"] == "BUY"].sum()),
            sell_dollar=("dollar_value", lambda s: s[tx.loc[s.index, "transaction_type"] == "SELL"].sum()),
        ).reindex(all_dates, fill_value=0)

        buy_roll = daily["buy_dollar"].rolling(TRAILING_WINDOW_DAYS, min_periods=1).sum()
        sell_roll = daily["sell_dollar"].rolling(TRAILING_WINDOW_DAYS, min_periods=1).sum()

        baseline_mean = buy_roll.rolling(BASELINE_WINDOW_DAYS, min_periods=10).mean()
        baseline_std = buy_roll.rolling(BASELINE_WINDOW_DAYS, min_periods=10).std().replace(0, np.nan)
        zscore = (buy_roll - baseline_mean) / baseline_std

        # distinct_buyers_5d needs an actual rolling DISTINCT count across
        # the window — summing per-day distinct counts double-counts an
        # insider who traded on two different days within the same window.
        buy_dates_sorted = buys["date"].sort_values().values
        buy_insiders_sorted = buys.sort_values("date")["insider_id"].values

        for i, d in enumerate(all_dates):
            window_start = d - pd.tseries.offsets.BDay(TRAILING_WINDOW_DAYS - 1)
            in_window = (buy_dates_sorted >= window_start) & (buy_dates_sorted <= d)
            distinct_buyers = len(set(buy_insiders_sorted[in_window])) if in_window.any() else 0

            total = buy_roll.iloc[i] + sell_roll.iloc[i]
            rows.append({
                "ticker": ticker,
                "date": d,
                "distinct_buyers_5d": distinct_buyers,
                "buy_dollar_volume_5d": buy_roll.iloc[i],
                "buy_sell_ratio_5d": (buy_roll.iloc[i] / total) if total > 0 else 0.5,
                "dollar_volume_zscore": zscore.iloc[i] if not np.isnan(zscore.iloc[i]) else 0.0,
            })

    return pd.DataFrame(rows)
