"""
backtest.py

Answers the actual question that matters: do the days a detector flags
tend to precede real price moves, more than a random day would?

Methodology:
  1. For every flagged (ticker, date), compute the forward return over
     `horizon_days` trading days.
  2. Compute the same forward-return distribution for a random sample of
     UNFLAGGED (ticker, date) pairs of the same size, as the null/control.
  3. Run an independent two-sample t-test (scipy.stats.ttest_ind) comparing
     the two groups' forward returns. This is the honest version of "does
     this signal actually mean anything" — eyeballing a few examples is
     not evidence, a significance test against a random control is.

This is intentionally simple (no transaction costs, no risk-adjustment,
no multiple-comparison correction) — it's a first-pass validation, not a
trading strategy. The README is explicit about this.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DEFAULT_CONFIG


def forward_return(prices: pd.DataFrame, ticker: str, date, horizon_days: int) -> float | None:
    series = prices[prices.ticker == ticker].sort_values("date").reset_index(drop=True)
    series["date"] = pd.to_datetime(series["date"])
    date = pd.to_datetime(date)
    idx_matches = series.index[series["date"] == date]
    if len(idx_matches) == 0:
        return None
    idx = idx_matches[0]
    if idx + horizon_days >= len(series):
        return None
    start_price = series.loc[idx, "close"]
    end_price = series.loc[idx + horizon_days, "close"]
    if start_price == 0:
        return None
    return (end_price - start_price) / start_price


def _collect_returns(rows: pd.DataFrame, prices: pd.DataFrame, horizon_days: int) -> list[float]:
    """Compute forward_return for every row, skipping ones too close to the data's edge."""
    returns = [forward_return(prices, row.ticker, row.date, horizon_days) for _, row in rows.iterrows()]
    return [r for r in returns if r is not None]


def backtest(detections: pd.DataFrame, prices: pd.DataFrame,
             horizon_days: int = DEFAULT_CONFIG.forward_return_horizon_days, seed: int = 42):
    """
    Returns a dict with flagged/control forward-return stats and a t-test
    result, plus the per-row forward returns for flagged days (useful for
    plotting).
    """
    flagged = detections[detections.is_anomaly]
    flagged_returns = _collect_returns(flagged, prices, horizon_days)

    # Control: random sample of unflagged (ticker, date) pairs, same size.
    unflagged = detections[~detections.is_anomaly]
    sample_size = min(len(flagged_returns) * 5, len(unflagged))  # more control points for a stabler estimate
    control_rows = unflagged.sample(n=sample_size, random_state=seed) if sample_size > 0 else unflagged
    control_returns = _collect_returns(control_rows, prices, horizon_days)

    flagged_returns = np.array(flagged_returns)
    control_returns = np.array(control_returns)

    result = {
        "n_flagged": len(flagged_returns),
        "n_control": len(control_returns),
        "flagged_mean_return": float(np.mean(flagged_returns)) if len(flagged_returns) else None,
        "control_mean_return": float(np.mean(control_returns)) if len(control_returns) else None,
        "flagged_returns": flagged_returns,
        "control_returns": control_returns,
    }

    if len(flagged_returns) >= 2 and len(control_returns) >= 2:
        t_stat, p_value = stats.ttest_ind(flagged_returns, control_returns, equal_var=False)
        result["t_stat"] = float(t_stat)
        result["p_value"] = float(p_value)
    else:
        result["t_stat"] = None
        result["p_value"] = None

    return result
