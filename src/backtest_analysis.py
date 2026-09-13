"""
backtest_analysis.py

Phase 3: a more rigorous evaluation layer built on top of backtest.py's
existing forward_return() (reused exactly as-is — the entry/exit
definition below is that same function's) and detect.py's leakage-safe
isolation_forest_detect_walkforward(). backtest.py's single-horizon
backtest() and its t-test are kept and still used by src/compare.py;
this module answers a broader set of questions about the same flagged
signals, not a replacement.

ENTRY/EXIT ASSUMPTIONS (read this before trusting any number below):

  - EXIT is unambiguous: `horizon_days` trading days after entry, at
    that day's close. Same as backtest.py.

  - ENTRY is where a subtlety lives that Phases 1-2 didn't surface.
    This project's `date` column is the Form 4 TRANSACTION date — the
    day the insider actually bought — not the FILING date. By law,
    Form 4 must be filed within 2 business days of the transaction,
    and filings are sometimes later still. A real trader could not
    have acted on the signal until it was actually FILED and public.
    Backtesting an entry at the transaction date's close (lag = 0,
    what backtest.py does) is therefore optimistic — it assumes
    information no real participant had yet.

    Every function below that computes forward returns accepts
    `entry_lag_days` (default 0, matching Phases 1-2's convention so
    old numbers stay reproducible) and shifts the effective entry date
    forward that many trading days before measuring the return.
    src/config.py's `entry_lag_days` (2, matching the legal filing
    deadline) is the realistic setting — the Phase 3 report runs both
    and shows the gap rather than picking one and hiding it.

BENCHMARK: there is no real market index in this project's universe —
tickers are synthetic. The benchmark used here is an EQUAL-WEIGHTED
AVERAGE forward return across every ticker in the dataset over the
same dates and horizon ("what a diversified basket of everything in
this universe would have returned"), not a claim about any real index.
This is the same honesty tradeoff Phase 2 made about "company size":
compute what the data actually supports, document what it doesn't.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import forward_return
from src.config import DEFAULT_CONFIG
from src.detect import isolation_forest_detect_walkforward, statistical_baseline
from src.evaluation import evaluate_detector


# ---------------------------------------------------------------------
# Forward returns (multi-horizon, lag-aware)
# ---------------------------------------------------------------------

def compute_forward_returns(
    detections: pd.DataFrame,
    prices: pd.DataFrame,
    horizons_days: tuple = DEFAULT_CONFIG.forward_return_horizons_days,
    entry_lag_days: int = 0,
) -> pd.DataFrame:
    """
    One row per flagged (ticker, date) in `detections`, with a forward
    return column per horizon in `horizons_days` (named
    `return_{h}d`). `entry_lag_days` shifts the effective entry date
    forward that many TRADING days (via each ticker's own price
    calendar) before measuring any horizon — see the module docstring.
    A signal too close to the end of the price history to reach a given
    horizon gets NaN for that horizon's column, not a dropped row.
    """
    if len(detections) == 0:
        return pd.DataFrame(columns=["ticker", "date", "entry_date"] + [f"return_{h}d" for h in horizons_days])

    flagged = detections[detections["is_anomaly"]].copy()
    flagged["date"] = pd.to_datetime(flagged["date"])
    if len(flagged) == 0:
        return pd.DataFrame(columns=["ticker", "date", "entry_date"] + [f"return_{h}d" for h in horizons_days])

    price_index = {
        ticker: grp.sort_values("date").reset_index(drop=True)
        for ticker, grp in prices.assign(date=pd.to_datetime(prices["date"])).groupby("ticker")
    }

    rows = []
    for _, sig in flagged.iterrows():
        series = price_index.get(sig.ticker)
        entry_date = sig.date
        if series is not None and entry_lag_days > 0:
            idx_matches = series.index[series["date"] == sig.date]
            if len(idx_matches) and idx_matches[0] + entry_lag_days < len(series):
                entry_date = series.loc[idx_matches[0] + entry_lag_days, "date"]
            else:
                entry_date = None  # not enough price history past the signal to apply the lag at all

        row = {"ticker": sig.ticker, "date": sig.date, "entry_date": entry_date}
        for h in horizons_days:
            row[f"return_{h}d"] = (
                forward_return(prices, sig.ticker, entry_date, h) if entry_date is not None else None
            )
        rows.append(row)

    return pd.DataFrame(rows)


def compute_benchmark_returns(
    prices: pd.DataFrame,
    dates: pd.Series,
    horizons_days: tuple = DEFAULT_CONFIG.forward_return_horizons_days,
) -> pd.DataFrame:
    """
    For each unique date in `dates`, the equal-weighted average forward
    return across EVERY ticker in `prices` (not just flagged ones) —
    "what holding a diversified basket of this universe would have
    returned from that date." One row per unique date, columns
    `benchmark_return_{h}d`. See the module docstring for why this,
    not a real index.
    """
    unique_dates = pd.to_datetime(pd.Series(dates).unique())
    tickers = prices["ticker"].unique()

    rows = []
    for d in unique_dates:
        row = {"date": d}
        for h in horizons_days:
            per_ticker = [r for t in tickers if (r := forward_return(prices, t, d, h)) is not None]
            row[f"benchmark_return_{h}d"] = float(np.mean(per_ticker)) if per_ticker else None
        rows.append(row)
    return pd.DataFrame(rows)


def compute_excess_returns(forward_returns: pd.DataFrame, benchmark_returns: pd.DataFrame,
                            horizons_days: tuple = DEFAULT_CONFIG.forward_return_horizons_days) -> pd.DataFrame:
    """Left-joins benchmark returns onto forward_returns by signal date
    and adds `excess_return_{h}d` = signal return - benchmark return,
    for every horizon present in both."""
    merged = forward_returns.merge(benchmark_returns, on="date", how="left")
    for h in horizons_days:
        sig_col, bench_col = f"return_{h}d", f"benchmark_return_{h}d"
        if sig_col in merged.columns and bench_col in merged.columns:
            merged[f"excess_return_{h}d"] = merged[sig_col] - merged[bench_col]
    return merged


# ---------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------

def apply_transaction_costs(
    returns: np.ndarray,
    transaction_cost_bps: float = DEFAULT_CONFIG.transaction_cost_bps,
    slippage_bps: float = DEFAULT_CONFIG.slippage_bps,
) -> np.ndarray:
    """
    Subtracts a round-trip drag from every return: transaction cost and
    slippage are each charged once on entry and once on exit, so the
    total deducted is 2 * (transaction_cost_bps + slippage_bps) / 10000.
    Applied uniformly per trade — a simplification (real slippage
    scales with trade size and liquidity, which this project has no
    data to model) stated here rather than hidden in the number.
    """
    round_trip_drag = 2 * (transaction_cost_bps + slippage_bps) / 10_000
    return np.asarray(returns, dtype=float) - round_trip_drag


# ---------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------

def compute_performance_metrics(returns: np.ndarray, periods_per_year: float = None,
                                 horizon_days: int = DEFAULT_CONFIG.forward_return_horizon_days) -> dict:
    """
    Standard per-trade performance stats for an array of (already
    horizon-matched) returns. `periods_per_year` defaults to
    252 / horizon_days — treating the holding period as the unit of
    time for annualizing Sharpe/volatility. This assumes trades are
    taken back-to-back and don't overlap; insider-signal trades often
    DO overlap in real time (several tickers flagged in the same week),
    so the annualized figures here should be read as a standard,
    comparable convention, not a claim about a literal continuously-
    invested portfolio — see "Honest limitations" wherever this is reported.

    Returns n=0 and None for every other field if there are no returns
    at all (empty input), rather than raising or fabricating zeros.

    ** cumulative_return and max_drawdown assume trades are taken
    sequentially with full capital reinvested between them. Insider-
    signal trades routinely overlap in real calendar time (the same
    or different tickers flagged days apart, each with a multi-day
    holding period) — for an overlapping sample these two numbers can
    compound into large-looking figures that were never achievable
    with one unit of capital. mean/median/hit_rate/Sharpe/volatility
    do NOT have this problem (each is a plain per-trade statistic) and
    should be treated as the primary evidence; cumulative_return and
    max_drawdown are reported for completeness, not as headline results. **
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n = len(returns)

    if n == 0:
        return {"n": 0, "mean_return": None, "median_return": None, "hit_rate": None,
                "cumulative_return": None, "max_drawdown": None, "sharpe_ratio": None,
                "volatility": None, "best_trade": None, "worst_trade": None}

    if periods_per_year is None:
        periods_per_year = 252 / horizon_days

    # Cumulative return / drawdown: compound the trades IN CHRONOLOGICAL
    # ORDER as if taken one after another with equal sizing — the
    # standard event-study convention for a set of discrete trades, not
    # a simulation of simultaneously-held overlapping positions (this
    # project's signals frequently DO overlap in calendar time).
    equity_curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=1)) if n > 1 else 0.0

    return {
        "n": n,
        "mean_return": mean_return,
        "median_return": float(np.median(returns)),
        "hit_rate": float(np.mean(returns > 0)),
        "cumulative_return": float(equity_curve[-1] - 1),
        "max_drawdown": float(drawdowns.min()),
        "sharpe_ratio": float((mean_return / std_return) * np.sqrt(periods_per_year)) if std_return > 0 else None,
        "volatility": float(std_return * np.sqrt(periods_per_year)),
        "best_trade": float(np.max(returns)),
        "worst_trade": float(np.min(returns)),
    }


# ---------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------

def significance_test(sample_returns: np.ndarray, control_returns: np.ndarray, confidence: float = 0.95) -> dict:
    """
    Independent two-sample t-test (Welch's, unequal variance assumed —
    same choice backtest.py already makes) between sample_returns
    (e.g. flagged-signal returns) and control_returns (e.g. random
    unflagged days or the benchmark), plus a confidence interval for
    the MEAN of sample_returns via the t-distribution.

    Reports sample sizes and honestly returns is_significant=False
    (never fabricates significance) whenever either sample is too
    small (< 2 observations) for the test to mean anything, with
    p_value=None to make that explicit rather than reporting a
    borderline or misleading number.
    """
    sample_returns = np.asarray(sample_returns, dtype=float)
    sample_returns = sample_returns[~np.isnan(sample_returns)]
    control_returns = np.asarray(control_returns, dtype=float)
    control_returns = control_returns[~np.isnan(control_returns)]

    result = {
        "n_sample": len(sample_returns),
        "n_control": len(control_returns),
        "mean_sample": float(np.mean(sample_returns)) if len(sample_returns) else None,
        "mean_control": float(np.mean(control_returns)) if len(control_returns) else None,
        "t_stat": None, "p_value": None,
        "ci_low": None, "ci_high": None, "confidence": confidence,
        "is_significant": False,
    }

    if len(sample_returns) >= 2:
        sem = stats.sem(sample_returns)
        if sem > 0:
            margin = sem * stats.t.ppf((1 + confidence) / 2, df=len(sample_returns) - 1)
            result["ci_low"] = float(result["mean_sample"] - margin)
            result["ci_high"] = float(result["mean_sample"] + margin)

    if len(sample_returns) >= 2 and len(control_returns) >= 2:
        t_stat, p_value = stats.ttest_ind(sample_returns, control_returns, equal_var=False)
        result["t_stat"] = float(t_stat)
        result["p_value"] = float(p_value)
        result["is_significant"] = bool(p_value < (1 - confidence))

    return result


# ---------------------------------------------------------------------
# Temporal train/test split
# ---------------------------------------------------------------------

def split_by_date(dates: pd.Series, development_fraction: float = DEFAULT_CONFIG.development_period_fraction):
    """
    Returns `split_date`: every date <= split_date is the DEVELOPMENT
    period, every date after it is OUT-OF-SAMPLE. Split point is the
    `development_fraction` quantile of the UNIQUE sorted dates —
    chronological, never a random shuffle (shuffling would let
    out-of-sample rows leak information into development via shared
    tickers' nearby dates, defeating the point of the split).
    """
    unique_dates = pd.to_datetime(pd.Series(dates).unique())
    unique_dates = np.sort(unique_dates)
    if len(unique_dates) == 0:
        raise ValueError("no dates to split")
    cutoff_idx = max(0, min(len(unique_dates) - 1, int(len(unique_dates) * development_fraction) - 1))
    return pd.Timestamp(unique_dates[cutoff_idx])


# ---------------------------------------------------------------------
# Out-of-sample backtest (ties the pieces above together)
# ---------------------------------------------------------------------

def run_out_of_sample_backtest(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    ground_truth: pd.DataFrame,
    development_fraction: float = DEFAULT_CONFIG.development_period_fraction,
    entry_lag_days: int = DEFAULT_CONFIG.entry_lag_days,
    horizon_days: int = DEFAULT_CONFIG.forward_return_horizon_days,
    transaction_cost_bps: float = DEFAULT_CONFIG.transaction_cost_bps,
    slippage_bps: float = DEFAULT_CONFIG.slippage_bps,
    seed: int = 42,
) -> dict:
    """
    Chronological development/out-of-sample split (see split_by_date):
    rows dated at or before the split are DEVELOPMENT, everything after
    is OUT-OF-SAMPLE. This project does not currently tune any detector
    parameter on data — detect.py's thresholds are fixed judgment
    calls, documented as such, not fit — so there is nothing for the
    development period to select today. The split is implemented and
    enforced anyway, for two honest reasons: (1) it's the discipline
    that has to already be in place the moment a future phase DOES
    start tuning something, and (2) it's what makes "out-of-sample"
    below mean something real rather than a label — every number this
    function reports is computed exclusively on rows dated after the
    split.

    The walk-forward Isolation Forest is still fit on the full feature
    history available up to each scored date (which naturally includes
    development-period rows as training history — exactly how it would
    behave in production, using everything known so far) but is never
    SCORED on a development-period date, and never trained on an
    out-of-sample date it hasn't reached yet chronologically. See
    detect.isolation_forest_detect_walkforward for that mechanism.

    Returns a dict: {"split_date", "n_development_rows",
    "n_out_of_sample_rows", "detectors": {label: {...}}} where each
    detector's dict has n_flagged, forward-return performance metrics
    at both entry_lag_days=0 and the realistic lag (with and without
    transaction costs), a significance test against random OOS control
    days, and ground-truth precision/recall/F1 restricted to OOS rows.
    """
    split_date = split_by_date(features["date"], development_fraction)
    features = features.copy()
    features["date"] = pd.to_datetime(features["date"])
    oos_prices = prices

    detectors = {
        "Statistical baseline": statistical_baseline(features),
        "Isolation Forest (walk-forward)": isolation_forest_detect_walkforward(features, seed=seed),
    }

    results = {
        "split_date": split_date,
        "n_development_rows": int((features["date"] <= split_date).sum()),
        "n_out_of_sample_rows": int((features["date"] > split_date).sum()),
        "detectors": {},
    }

    for label, detections in detectors.items():
        detections = detections.copy()
        detections["date"] = pd.to_datetime(detections["date"])
        oos = detections[detections["date"] > split_date]
        oos_flagged = oos[oos["is_anomaly"]]
        oos_unflagged = oos[(~oos["is_anomaly"]) & oos["score"].notna()]

        fr_lag0 = compute_forward_returns(oos, oos_prices, entry_lag_days=0)
        fr_realistic = compute_forward_returns(oos, oos_prices, entry_lag_days=entry_lag_days)

        control_sample_size = min(len(fr_lag0) * 5, len(oos_unflagged)) if len(fr_lag0) else 0
        control_rows = (
            oos_unflagged.sample(n=control_sample_size, random_state=seed) if control_sample_size > 0 else oos_unflagged
        )
        control_returns = compute_forward_returns(
            control_rows.assign(is_anomaly=True), oos_prices, horizons_days=(horizon_days,), entry_lag_days=0
        )[f"return_{horizon_days}d"].dropna().values

        headline_col = f"return_{horizon_days}d"
        headline_returns_lag0 = fr_lag0[headline_col].dropna().values if len(fr_lag0) else np.array([])
        headline_returns_realistic = fr_realistic[headline_col].dropna().values if len(fr_realistic) else np.array([])
        headline_returns_after_costs = apply_transaction_costs(
            headline_returns_realistic, transaction_cost_bps, slippage_bps
        )

        results["detectors"][label] = {
            "n_flagged_oos": int(len(oos_flagged)),
            "forward_returns_all_horizons_lag0": fr_lag0,
            "forward_returns_all_horizons_realistic": fr_realistic,
            "metrics_lag0": compute_performance_metrics(headline_returns_lag0, horizon_days=horizon_days),
            "metrics_realistic": compute_performance_metrics(headline_returns_realistic, horizon_days=horizon_days),
            "metrics_realistic_after_costs": compute_performance_metrics(
                headline_returns_after_costs, horizon_days=horizon_days
            ),
            "significance_vs_control": significance_test(headline_returns_realistic, control_returns),
            "ground_truth_eval": evaluate_detector(oos, ground_truth),
        }

    return results
