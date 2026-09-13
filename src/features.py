"""
features.py

Turns raw Form 4 transactions into a per-(ticker, day) feature table an
anomaly detector can use. The key insight: a single insider buying
isn't unusual — insiders buy their own stock all the time for mundane
reasons. What's unusual is CLUSTERING: several DIFFERENT insiders at the
same company buying independently within a tight window, especially
when it's unusually large for them, unusually large for the company,
senior people are involved, or it follows a price drop rather than a
run-up. That's the pattern this project looks for, not "did an insider
buy."

WHAT INFORMATION IS AVAILABLE AT SIGNAL TIME (no look-ahead leakage):
Every feature below for date `d` is computed using only transactions
and prices dated on or before `d`. pandas' `.rolling()` and
`.pct_change()` are backward-looking by construction; the two features
that aren't simple rolling windows (insider_relative_size_zscore_5d,
role_weighted_buy_value_5d) are built the same way, transaction by
transaction, in date order, using only each insider's OWN prior trades.
None of this looks at day d+1 or later. See test/test_features_v2.py's
TestNoLeakage for a test that actively checks this (perturbing future
rows doesn't change a past row's feature values).

Features computed per (ticker, date), using a trailing window
(`trailing_window_days`, default 5 — this is what "cluster window"
means throughout this project: the span of days a set of purchases has
to fall within to count as one clustered episode):

  KEPT FROM THE ORIGINAL DETECTOR:
  - distinct_buyers_5d: how many different insiders bought in the
    window (a genuine rolling DISTINCT count, not a sum of daily counts
    — an insider trading twice shouldn't count as two people). This is
    the core cluster signal: one person buying is routine, several
    different people buying independently is not.
  - buy_dollar_volume_5d: total $ value of insider buys in the window.
  - buy_sell_ratio_5d: buy $ / (buy $ + sell $) — skews toward 1 when
    insiders are unusually one-sided (buy/sell imbalance).
  - dollar_volume_zscore: buy_dollar_volume_5d expressed as a z-score
    against THAT TICKER's own trailing 60-day history (`baseline_window_days`)
    — lets a normally-quiet stock and a normally-active one be compared
    fairly, and doubles as this project's "abnormal purchase volume"
    and "historical insider activity" signal: a company's own baseline
    already encodes how much insider trading is normal for it.

  NEW IN PHASE 2:
  - n_purchases_5d: total number of BUY transactions in the window
    (distinct from distinct_buyers_5d — 3 buyers making 10 trades looks
    different from 3 buyers making 3 trades).
  - purchases_per_buyer_5d: n_purchases_5d / distinct_buyers_5d —
    purchase concentration. Several trades from the same small group of
    people reads differently than the same trade count spread widely.
  - insider_relative_size_zscore_5d: how large the largest buy in the
    window is relative to THAT SPECIFIC INSIDER's own trailing trade
    history (not the company's) — a $50k purchase is routine for
    someone who trades $50k regularly and remarkable for someone who's
    never bought more than $2k. Needs >= `insider_history_min_trades`
    PRIOR trades by that insider before it's trusted; below that it's
    0.0 (not "no signal" — genuinely not enough history to judge).
  - role_weighted_buy_value_5d: buy_dollar_volume_5d, but each
    transaction is weighted by the buyer's seniority (CEO/CFO highest,
    then President/COO, then Director, then everyone else) before
    summing — see `src.config.PipelineConfig.role_weights`. A CFO's
    purchase is prima facie more informative than an unspecified
    filer's; this is a simple, inspectable weighting, not a fitted one.
    Falls back to unweighted (all weight 1.0, identical to
    buy_dollar_volume_5d) if the input has no insider_title column.
  - pre_signal_return_10d: the ticker's own stock return over the
    `pre_signal_return_window_days` (default 10) trading days leading
    into date d, using only prices through d. Captures whether insiders
    are buying after a price drop (a "buying the dip" pattern some
    research associates with informed buying) versus a run-up. 0.0 if
    no `prices` argument was given, or price history doesn't reach back
    far enough.

NOT IMPLEMENTED: "purchase value relative to company size." This would
need market capitalization or shares outstanding, which aren't in
either data source this project has (`generate_synthetic_data.py`'s
close price, or the real Form 4 schema in `src/edgar/`) — a closing
price alone isn't a size proxy without a share count to multiply it by,
and fabricating one from price level alone wouldn't make financial
sense (a $500 stock isn't a bigger company than a $50 one). The closest
available substitute is dollar_volume_zscore, which normalizes against
each ticker's OWN historical scale rather than a cross-sectional size
measure. Documented here rather than faked.
"""

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG

FEATURE_COLUMNS = [
    "ticker", "date",
    "distinct_buyers_5d", "buy_dollar_volume_5d", "buy_sell_ratio_5d", "dollar_volume_zscore",
    "n_purchases_5d", "purchases_per_buyer_5d",
    "insider_relative_size_zscore_5d", "role_weighted_buy_value_5d", "pre_signal_return_10d",
]

# The numeric feature columns a detector should actually train/score on
# (excludes the ticker/date identifier columns). Kept here, not just in
# detect.py, so features.py is the one place that defines "what a
# feature vector for this problem looks like."
NUMERIC_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("ticker", "date")]


def _role_weight(title, role_weights: dict) -> float:
    """First matching substring (case-insensitive) in role_weights wins;
    an unrecognized or missing title gets the neutral weight 1.0."""
    if not title:
        return 1.0
    title_lower = str(title).lower()
    for substring, weight in role_weights.items():
        if substring in title_lower:
            return weight
    return 1.0


def build_features(
    transactions: pd.DataFrame,
    prices: pd.DataFrame = None,
    trailing_window_days: int = DEFAULT_CONFIG.trailing_window_days,
    baseline_window_days: int = DEFAULT_CONFIG.baseline_window_days,
    insider_history_min_trades: int = DEFAULT_CONFIG.insider_history_min_trades,
    pre_signal_return_window_days: int = DEFAULT_CONFIG.pre_signal_return_window_days,
    role_weights: dict = None,
) -> pd.DataFrame:
    """
    `prices` is optional — pass it to get a real pre_signal_return_10d;
    without it, that column is 0.0 for every row (neutral, not missing,
    so the output schema is identical either way and a detector doesn't
    need to know which case it's in).
    """
    if role_weights is None:
        role_weights = DEFAULT_CONFIG.role_weights

    if transactions.empty:
        # No transactions at all — return an empty, correctly-shaped table
        # rather than crashing on pd.bdate_range(NaT, NaT). A real pipeline
        # can hit this on a ticker with no filings in the lookback window.
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    transactions = transactions.copy()
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["dollar_value"] = transactions["shares"] * transactions["price"]
    has_titles = "insider_title" in transactions.columns

    all_dates = pd.bdate_range(transactions["date"].min(), transactions["date"].max())
    tickers = transactions["ticker"].unique()

    rows = []
    for ticker in tickers:
        tx = transactions[transactions["ticker"] == ticker]
        buys = tx[tx.transaction_type == "BUY"].sort_values("date")

        # Daily $ aggregates via simple reindexed groupby — cheap and clear.
        daily = tx.groupby("date").agg(
            buy_dollar=("dollar_value", lambda s: s[tx.loc[s.index, "transaction_type"] == "BUY"].sum()),
            sell_dollar=("dollar_value", lambda s: s[tx.loc[s.index, "transaction_type"] == "SELL"].sum()),
        ).reindex(all_dates, fill_value=0)

        buy_roll = daily["buy_dollar"].rolling(trailing_window_days, min_periods=1).sum()
        sell_roll = daily["sell_dollar"].rolling(trailing_window_days, min_periods=1).sum()

        baseline_mean = buy_roll.rolling(baseline_window_days, min_periods=10).mean()
        baseline_std = buy_roll.rolling(baseline_window_days, min_periods=10).std().replace(0, np.nan)
        zscore = (buy_roll - baseline_mean) / baseline_std

        # ---- per-transaction arrays used by the rolling-window lookups
        # below (distinct buyers, purchase count, role weighting, and
        # each insider's own trailing history) ----
        buy_dates_sorted = buys["date"].values
        buy_insiders_sorted = buys["insider_id"].values
        buy_dollar_sorted = buys["dollar_value"].values
        buy_titles_sorted = buys["insider_title"].values if has_titles else np.full(len(buys), None)
        buy_weight_sorted = np.array([_role_weight(t, role_weights) for t in buy_titles_sorted])

        # insider_relative_size_zscore: for each buy, compare its $ value
        # to THAT insider's own PRIOR trades only (strictly earlier dates
        # within this ticker) — a per-transaction, point-in-time
        # computation, not a rolling window over the whole series.
        insider_history: dict = {}
        per_tx_relative_zscore = np.zeros(len(buys))
        for i, (insider_id, dollar_value) in enumerate(zip(buy_insiders_sorted, buy_dollar_sorted)):
            hist = insider_history.get(insider_id, [])
            if len(hist) >= insider_history_min_trades:
                hist_mean = np.mean(hist)
                hist_std = np.std(hist)
                if hist_std > 0:
                    per_tx_relative_zscore[i] = (dollar_value - hist_mean) / hist_std
                elif dollar_value != hist_mean:
                    # Perfectly consistent history (std == 0) but this
                    # trade differs from it — genuinely unusual for this
                    # insider, not "no signal." A real std-based z-score
                    # would be infinite here; 5.0/-5.0 is a capped stand-in
                    # consistent with how this feature is capped elsewhere
                    # (see signal_score.py's saturation value for it).
                    per_tx_relative_zscore[i] = 5.0 if dollar_value > hist_mean else -5.0
            insider_history.setdefault(insider_id, []).append(dollar_value)

        # ---- pre_signal_return_10d: backward-looking return, only if
        # price history was supplied. Computed on the price series' OWN
        # date range, not `all_dates` (which comes from the
        # transactions and can be much narrower/sparser than the price
        # history available to look back over — e.g. a ticker with only
        # a few transactions near the end of a long price history).
        if prices is not None:
            ticker_prices = prices[prices["ticker"] == ticker].copy()
            ticker_prices["date"] = pd.to_datetime(ticker_prices["date"])
            price_series = ticker_prices.sort_values("date").set_index("date")["close"]
            pre_signal_return_full = price_series.pct_change(periods=pre_signal_return_window_days)
            pre_signal_return = pre_signal_return_full.reindex(all_dates).ffill()
        else:
            pre_signal_return = pd.Series(0.0, index=all_dates)

        for i, d in enumerate(all_dates):
            window_start = d - pd.tseries.offsets.BDay(trailing_window_days - 1)
            in_window = (buy_dates_sorted >= window_start) & (buy_dates_sorted <= d)
            n_in_window = int(in_window.sum())
            distinct_buyers = len(set(buy_insiders_sorted[in_window])) if n_in_window else 0

            total = buy_roll.iloc[i] + sell_roll.iloc[i]
            return_val = pre_signal_return.iloc[i]

            rows.append({
                "ticker": ticker,
                "date": d,
                "distinct_buyers_5d": distinct_buyers,
                "buy_dollar_volume_5d": buy_roll.iloc[i],
                "buy_sell_ratio_5d": (buy_roll.iloc[i] / total) if total > 0 else 0.5,
                "dollar_volume_zscore": zscore.iloc[i] if not np.isnan(zscore.iloc[i]) else 0.0,
                "n_purchases_5d": n_in_window,
                "purchases_per_buyer_5d": (n_in_window / distinct_buyers) if distinct_buyers > 0 else 0.0,
                "insider_relative_size_zscore_5d": (
                    float(np.max(per_tx_relative_zscore[in_window])) if n_in_window else 0.0
                ),
                "role_weighted_buy_value_5d": (
                    float(np.sum(buy_dollar_sorted[in_window] * buy_weight_sorted[in_window])) if n_in_window else 0.0
                ),
                "pre_signal_return_10d": float(return_val) if not np.isnan(return_val) else 0.0,
            })

    return pd.DataFrame(rows)
