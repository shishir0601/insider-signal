"""
data_service.py

Bridges the dashboard to the existing analysis pipeline in src/. Builds
ONE synthetic demo dataset per process (cached at first request) and
exposes a single function, analyze_ticker(ticker), returning everything
the dashboard template needs.

ALL fully-analyzed data in this dashboard is synthetic DEMO DATA (see
README) unless explicitly marked "live." There is no real price-history
source in this project — Phase 1 added real Form 4 TRANSACTIONS via
src/edgar/, not prices — so a full feature+detect+backtest analysis is
only possible on the demo dataset. Typing a ticker not in the demo set
attempts a live, transactions-only SEC EDGAR lookup (only if
EDGAR_USER_AGENT is configured) and is clearly labeled as such; it
never claims to run detection on data this project has no prices for.
"""

import logging
import os

import pandas as pd

from src.compare import compare_detectors
from src.config import DEFAULT_CONFIG
from src.detect import isolation_forest_detect_walkforward, statistical_baseline
from src.edgar.client import EdgarClient
from src.edgar.ingest import fetch_transactions_for_ticker
from src.cleaning import clean_transactions
from src.features import build_features
from src.generate_synthetic_data import generate
from src.signal_score import compute_signal_score, explain_signal_score

logger = logging.getLogger(__name__)

DEMO_SEED = 42
CLUSTER_GAP_DAYS = 5  # a gap of more than this many days splits one flagged episode from another

_demo_cache = None


def get_demo_dataset() -> dict:
    global _demo_cache
    if _demo_cache is None:
        _demo_cache = _build_demo_dataset(DEMO_SEED)
    return _demo_cache


def _build_demo_dataset(seed: int) -> dict:
    tx, prices, ground_truth = generate(seed=seed)
    features = build_features(tx, prices=prices)

    stat_det = statistical_baseline(features)
    iso_det = isolation_forest_detect_walkforward(features, seed=seed)

    features = features.copy()
    features["signal_score"] = compute_signal_score(features)
    features["flagged_statistical"] = stat_det["is_anomaly"].values
    features["flagged_isolation_forest"] = iso_det["is_anomaly"].values
    features["flagged_any"] = features["flagged_statistical"] | features["flagged_isolation_forest"]
    features["date"] = pd.to_datetime(features["date"])

    combined_detections = features[["ticker", "date", "flagged_any", "signal_score"]].rename(
        columns={"flagged_any": "is_anomaly", "signal_score": "score"}
    )

    # Computed once, GLOBALLY (across all tickers), and cached — not
    # per-ticker: a single ticker has ~180 rows, below
    # walkforward_min_train_rows's default of 200, so a per-ticker walk
    # -forward fit would never train at all. This is also what makes
    # the comparison statistically meaningful in the first place (most
    # individual tickers have zero injected clusters to evaluate against).
    detector_comparison = compare_detectors(features, prices, ground_truth, seed=seed)

    return {
        "seed": seed,
        "transactions": tx,
        "prices": prices,
        "ground_truth": ground_truth,
        "features": features,
        "combined_detections": combined_detections,
        "detector_comparison": detector_comparison,
        "tickers": sorted(tx["ticker"].unique().tolist()),
    }


def analyze_ticker(ticker: str) -> dict:
    ticker = (ticker or "").strip().upper()
    demo = get_demo_dataset()
    if not ticker:
        ticker = demo["tickers"][0]

    if ticker in demo["tickers"]:
        return _build_demo_result(ticker, demo)

    live = _try_live_lookup(ticker)
    if live is not None:
        return live

    return {"found": False, "ticker": ticker, "is_demo": False, "demo_tickers": demo["tickers"]}


# ---------------------------------------------------------------------
# Demo (full analysis) path
# ---------------------------------------------------------------------

def _build_demo_result(ticker: str, demo: dict) -> dict:
    features = demo["features"]
    prices = demo["prices"]
    tx = demo["transactions"]

    t_features = features[features.ticker == ticker].sort_values("date")
    t_prices = prices[prices.ticker == ticker].sort_values("date")
    t_tx = tx[tx.ticker == ticker].copy()
    t_tx["date"] = pd.to_datetime(t_tx["date"])
    t_tx = t_tx.sort_values("date")

    latest = t_features.iloc[-1]
    flagged_rows = t_features[t_features.flagged_any]
    clusters = _group_into_clusters(flagged_rows)

    top_signal = None
    if len(flagged_rows):
        top_row = flagged_rows.sort_values(["signal_score", "date"], ascending=[False, False]).iloc[0]
        top_signal = _build_signal_detail(ticker, top_row, demo)

    recent_transactions = t_tx.tail(15).sort_values("date", ascending=False)

    return {
        "found": True,
        "is_demo": True,
        "ticker": ticker,
        "company_name": f"{ticker} — synthetic demo ticker",
        "recent_prices": t_prices.tail(15)[["date", "close"]],
        "recent_transactions": recent_transactions,
        "n_insiders_buying_recent": int(latest.distinct_buyers_5d),
        "total_purchase_value_recent": float(latest.buy_dollar_volume_5d),
        "n_clusters_detected": len(clusters),
        "clusters": clusters,
        "top_signal": top_signal,
        "demo_tickers": demo["tickers"],
        # raw slices for chart rendering (routes.py owns turning these into images)
        "_ticker_prices": t_prices,
        "_ticker_buys": t_tx[t_tx.transaction_type == "BUY"],
        "_ticker_signal_dates": flagged_rows["date"],
        "_demo": demo,
    }


def _group_into_clusters(flagged_rows: pd.DataFrame) -> list:
    if len(flagged_rows) == 0:
        return []
    dates = sorted(pd.to_datetime(flagged_rows["date"]).unique())
    episodes = []
    start = prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days > CLUSTER_GAP_DAYS:
            episodes.append((start, prev))
            start = d
        prev = d
    episodes.append((start, prev))

    enriched = []
    for start, end in episodes:
        window = flagged_rows[
            (pd.to_datetime(flagged_rows["date"]) >= start) & (pd.to_datetime(flagged_rows["date"]) <= end)
        ]
        detectors = []
        if window["flagged_statistical"].any():
            detectors.append("Statistical baseline")
        if window["flagged_isolation_forest"].any():
            detectors.append("Isolation Forest")
        enriched.append({
            "start": start, "end": end,
            "n_days": (end - start).days + 1,
            "peak_score": float(window["signal_score"].max()),
            "detectors": detectors,
        })
    return enriched


def _build_signal_detail(ticker: str, row: pd.Series, demo: dict) -> dict:
    tx = demo["transactions"]
    window_end = pd.to_datetime(row["date"])
    window_start = window_end - pd.tseries.offsets.BDay(DEFAULT_CONFIG.trailing_window_days - 1)

    t_tx = tx[tx.ticker == ticker].copy()
    t_tx["date"] = pd.to_datetime(t_tx["date"])
    window_buys = t_tx[
        (t_tx.transaction_type == "BUY") & (t_tx.date >= window_start) & (t_tx.date <= window_end)
    ]

    insiders = []
    for insider_id, grp in window_buys.groupby("insider_id"):
        insiders.append({
            "insider_id": insider_id,
            "title": grp["insider_title"].iloc[0] if "insider_title" in grp.columns and len(grp) else "",
            "total_value": float((grp["shares"] * grp["price"]).sum()),
            "n_purchases": int(len(grp)),
        })
    insiders.sort(key=lambda x: -x["total_value"])

    detectors_triggered = []
    if row.get("flagged_statistical"):
        detectors_triggered.append("Statistical baseline")
    if row.get("flagged_isolation_forest"):
        detectors_triggered.append("Isolation Forest (walk-forward)")

    return {
        "date": row["date"],
        "score": float(row["signal_score"]),
        "score_breakdown": explain_signal_score(row),
        "detectors_triggered": detectors_triggered,
        "insiders": insiders,
        "cluster_window": (window_start, window_end),
        "cluster_window_days": DEFAULT_CONFIG.trailing_window_days,
        "distinct_buyers": int(row["distinct_buyers_5d"]),
        "total_purchase_value": float(row["buy_dollar_volume_5d"]),
        "dollar_volume_zscore": float(row["dollar_volume_zscore"]),
        "explanation": _build_explanation(row),
    }


def _build_explanation(row: pd.Series) -> str:
    """The "why was this flagged" sentence, built entirely from this
    row's own actual feature values — never a canned string."""
    n = int(row["distinct_buyers_5d"])
    total_value = row["buy_dollar_volume_5d"]
    z = row["dollar_volume_zscore"]
    window_days = DEFAULT_CONFIG.trailing_window_days

    sentence = (
        f"{n} distinct insider{'s' if n != 1 else ''} purchased shares within {window_days} days, "
        f"with total purchases of ${total_value:,.0f}. This is {z:.1f} standard deviations above "
        f"this company's own historical purchase activity."
    )

    extras = []
    if row.get("role_weighted_buy_value_5d", 0) > total_value * 1.15:
        extras.append("Buying was concentrated among senior insiders (e.g. CEO/CFO), which this system weights more heavily.")
    if row.get("purchases_per_buyer_5d", 1) > 1.4:
        extras.append("Several of these insiders bought more than once within the window.")
    if row.get("pre_signal_return_10d", 0) < -0.03:
        extras.append(f"This follows a {row['pre_signal_return_10d'] * 100:.1f}% price decline over the prior 10 trading days.")

    return sentence + ((" " + " ".join(extras)) if extras else "")


# ---------------------------------------------------------------------
# Live (transactions-only) path
# ---------------------------------------------------------------------

def _try_live_lookup(ticker: str):
    """Returns a result dict for a live SEC EDGAR lookup, or None if
    live lookups aren't configured/available/successful — callers fall
    back to the "not found, try a demo ticker" response in that case.
    Any failure is logged server-side (network, auth, unexpected data)
    so a person running the dashboard can see why, even though the
    user-facing behavior is the same graceful fallback either way."""
    user_agent = os.environ.get("EDGAR_USER_AGENT")
    if not user_agent:
        return None

    try:
        client = EdgarClient(user_agent=user_agent)
        raw = fetch_transactions_for_ticker(client, ticker, limit=20)
    except Exception:
        logger.warning("Live EDGAR lookup failed for ticker %s", ticker, exc_info=True)
        return None

    if raw is None or len(raw) == 0:
        return None

    cleaned, _report = clean_transactions(raw)
    if len(cleaned) == 0:
        return None

    company_name = ticker
    if "issuer_name" in raw.columns and len(raw):
        company_name = raw["issuer_name"].dropna().iloc[0] if raw["issuer_name"].notna().any() else ticker

    return {
        "found": True,
        "is_demo": False,
        "ticker": ticker,
        "company_name": company_name,
        "recent_transactions": cleaned.sort_values("date", ascending=False).head(15),
        "live_note": (
            "Live SEC EDGAR transactions shown above. This project has no real price-history "
            "source yet, so signal detection and backtesting below are not available for real "
            "tickers in this version — see the demo tickers for the full analysis."
        ),
        "demo_tickers": get_demo_dataset()["tickers"],
    }
