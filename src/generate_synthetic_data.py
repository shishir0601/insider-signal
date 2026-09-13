"""
generate_synthetic_data.py

IMPORTANT — READ THIS FIRST:
This project was built in an offline environment with no internet access,
so it cannot pull real SEC EDGAR Form 4 filings or real price data. This
module generates SYNTHETIC data that mirrors the exact structure of real
Form 4 insider transaction filings and daily OHLC price data, with a known
ground truth injected in (some tickers get a genuine "informed cluster"
of insider buying before a price jump; most don't).

Why this is still a legitimate ML exercise:
  - The detection pipeline (features.py, detect.py, backtest.py) is
    written against this schema and does not know which days are
    synthetic-injected anomalies — it has to find them the same way it
    would have to find them in real data.
  - Because we KNOW the ground truth here, we can validate that the
    pipeline actually works (recovers the injected signal) before ever
    pointing it at real data — this is good practice regardless of data
    source.
  - Swapping in real data means replacing this file only. Real Form 4
    filings are available in bulk from SEC EDGAR
    (https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany, or the
    full-text search / bulk data sets at https://www.sec.gov/data-research)
    and can be parsed into the same {ticker, insider_id, date, shares,
    price, transaction_type} schema used here.

Schema produced:
  transactions: ticker, insider_id, date, transaction_type (BUY/SELL), shares, price
  prices:       ticker, date, close
"""

import numpy as np
import pandas as pd

RNG_SEED = 42


def generate(
    n_tickers: int = 15,
    n_days: int = 180,
    n_insiders_per_ticker: int = 8,
    n_informed_clusters: int = 5,
    start_date: str = "2026-01-01",
    seed: int = RNG_SEED,
):
    """
    Returns (transactions_df, prices_df, ground_truth_df).

    ground_truth_df marks which (ticker, date) windows contain an injected
    informed-trading cluster — used only for validating the pipeline, never
    fed into the detector itself.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    tickers = [f"TCK{i:02d}" for i in range(n_tickers)]

    # ---- price series: geometric random walk per ticker ----
    prices = []
    price_paths = {}
    for t in tickers:
        daily_ret = rng.normal(0.0003, 0.015, size=n_days)
        path = 100 * np.cumprod(1 + daily_ret)
        price_paths[t] = path
        for d, p in zip(dates, path):
            prices.append({"ticker": t, "date": d, "close": round(p, 2)})
    prices_df = pd.DataFrame(prices)

    # ---- pick which tickers get an injected informed cluster ----
    informed_tickers = rng.choice(tickers, size=n_informed_clusters, replace=False)
    ground_truth = []

    transactions = []
    insider_ids = {t: [f"{t}-INS{i}" for i in range(n_insiders_per_ticker)] for t in tickers}

    # ---- insider roles (Phase 2: needed for role-weighted features) ----
    # Deliberately drawn from a SEPARATE RNG stream, not `rng` above: this
    # was added after the price/cluster/transaction generation logic
    # already existed and was pinned by a regression test
    # (test_default_seed_output_unchanged_by_bounds_fix) — inserting new
    # draws into the middle of the existing `rng` sequence would shift
    # every draw after it and silently change which tickers get injected
    # clusters, the price paths, and every transaction for a given seed.
    # An independent stream means role assignment is still fully
    # deterministic (same seed -> same roles) without perturbing any
    # existing output.
    role_rng = np.random.default_rng(seed + 1_000_003)
    role_fill_template = ["Director", "VP", "Officer", "10% Owner"]
    insider_roles = {}
    for t in tickers:
        ids = insider_ids[t]
        pool = []
        if len(ids) >= 1:
            pool.append("CEO")
        if len(ids) >= 2:
            pool.append("CFO")
        i = 0
        while len(pool) < len(ids):
            pool.append(role_fill_template[i % len(role_fill_template)])
            i += 1
        role_rng.shuffle(pool)
        for insider_id, role in zip(ids, pool):
            insider_roles[insider_id] = role

    for t in tickers:
        # ---- baseline noise trading: sparse, uncorrelated ----
        for insider in insider_ids[t]:
            n_trades = rng.poisson(2.5)
            for _ in range(n_trades):
                d = rng.choice(dates)
                shares = int(rng.integers(100, 5000))
                ttype = rng.choice(["BUY", "SELL"], p=[0.45, 0.55])
                price_on_day = price_paths[t][list(dates).index(d)]
                transactions.append({
                    "ticker": t, "insider_id": insider, "date": d,
                    "transaction_type": ttype, "shares": shares,
                    "price": round(price_on_day, 2),
                    "insider_title": insider_roles[insider],
                })

        if t in informed_tickers:
            # ---- inject an informed cluster: several insiders buy in a
            # tight window, followed by a real price jump a few days later.
            cluster_start_idx = int(rng.integers(20, n_days - 30))
            cluster_window = dates[cluster_start_idx: cluster_start_idx + 4]
            jump_idx = cluster_start_idx + rng.integers(6, 12)

            n_informed_insiders = rng.integers(4, n_insiders_per_ticker)
            informed_insiders = rng.choice(insider_ids[t], size=n_informed_insiders, replace=False)
            for insider in informed_insiders:
                d = rng.choice(cluster_window)
                shares = int(rng.integers(3000, 15000))  # notably larger than noise trades
                price_on_day = price_paths[t][list(dates).index(d)]
                transactions.append({
                    "ticker": t, "insider_id": insider, "date": d,
                    "transaction_type": "BUY", "shares": shares,
                    "price": round(price_on_day, 2),
                    "insider_title": insider_roles[insider],
                })

            # bake in the forward price jump so the backtest has something real to find
            if jump_idx < n_days:
                jump_size = rng.uniform(0.08, 0.22)
                price_paths[t][jump_idx:] *= (1 + jump_size)
                for i, d in enumerate(dates):
                    prices_df.loc[(prices_df.ticker == t) & (prices_df.date == d), "close"] = round(price_paths[t][i], 2)

            ground_truth.append({
                "ticker": t,
                "cluster_start": cluster_window[0],
                "cluster_end": cluster_window[-1],
                "jump_date": dates[jump_idx] if jump_idx < n_days else None,
            })

    transactions_df = pd.DataFrame(transactions).sort_values(["ticker", "date"]).reset_index(drop=True)
    ground_truth_df = pd.DataFrame(ground_truth)
    return transactions_df, prices_df, ground_truth_df


if __name__ == "__main__":
    tx, px, gt = generate()
    tx.to_csv("data/transactions.csv", index=False)
    px.to_csv("data/prices.csv", index=False)
    gt.to_csv("data/ground_truth.csv", index=False)
    print(f"Generated {len(tx)} transactions across {tx.ticker.nunique()} tickers")
    print(f"Injected {len(gt)} informed clusters (ground truth, for validation only):")
    print(gt)
