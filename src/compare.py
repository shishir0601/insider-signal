"""
compare.py

Runs all three approaches in detect.py — random baseline, statistical
baseline, Isolation Forest — on the SAME features, prices, and ground
truth, through the SAME two evaluation lenses, so they can be read side
by side instead of each having been checked on its own terms:

  - src.backtest.backtest(): does this matter economically? (forward
    returns of flagged days vs. a random control, with a significance test)
  - src.evaluation.evaluate_detector(): is this actually finding the
    right days? (precision/recall/F1/PR-AUC/ROC-AUC against the known
    injected clusters — only possible because this is synthetic
    validation data)

Isolation Forest is compared in its leakage-safe walk-forward form
(isolation_forest_detect_walkforward), not the faster full-dataset fit
— a comparison meant to answer "which of these would actually work" has
to use the leakage-safe version of each detector it includes; see
detect.py for why the two Isolation Forest variants exist.

Usage:
    from src.generate_synthetic_data import generate
    from src.features import build_features
    from src.compare import compare_detectors

    tx, prices, ground_truth = generate()
    features = build_features(tx, prices=prices)
    results = compare_detectors(features, prices, ground_truth)
    print(results)  # a DataFrame, one row per detector
"""

import pandas as pd

from src.backtest import backtest
from src.detect import (
    isolation_forest_detect_walkforward,
    random_baseline_detect,
    statistical_baseline,
)
from src.evaluation import evaluate_detector

DETECTORS = {
    "Random baseline": random_baseline_detect,
    "Statistical baseline": statistical_baseline,
    "Isolation Forest (walk-forward)": isolation_forest_detect_walkforward,
}


def compare_detectors(features: pd.DataFrame, prices: pd.DataFrame, ground_truth: pd.DataFrame,
                       horizon_days: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per detector and columns for both
    evaluation lenses: n_flagged, flagged_pct, forward-return backtest
    stats (mean_return_flagged, mean_return_control, t_stat, p_value),
    and ground-truth classification stats (precision, recall, f1,
    pr_auc, roc_auc). Detectors that flagged nothing, or that have no
    scored rows (e.g. an all-NaN warm-up period), report None rather
    than a fabricated 0 for the metrics that need at least one flag.
    """
    rows = []
    for label, detect_fn in DETECTORS.items():
        detections = detect_fn(features, seed=seed) if label != "Statistical baseline" else detect_fn(features)
        bt = backtest(detections, prices, horizon_days=horizon_days, seed=seed)
        ev = evaluate_detector(detections, ground_truth)

        n_scored = int(detections["score"].notna().sum())
        n_flagged = int(detections["is_anomaly"].sum())
        rows.append({
            "detector": label,
            "n_scored_rows": n_scored,
            "n_flagged": n_flagged,
            "flagged_pct": round(100 * n_flagged / n_scored, 2) if n_scored else None,
            "mean_return_flagged": bt["flagged_mean_return"],
            "mean_return_control": bt["control_mean_return"],
            "t_stat": bt["t_stat"],
            "p_value": bt["p_value"],
            "precision": ev["precision"],
            "recall": ev["recall"],
            "f1": ev["f1"],
            "pr_auc": ev["pr_auc"],
            "roc_auc": ev["roc_auc"],
        })

    return pd.DataFrame(rows).set_index("detector")
