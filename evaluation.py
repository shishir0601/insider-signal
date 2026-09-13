"""
evaluation.py

The backtest (backtest.py) answers "does this matter economically" — do
flagged days show better forward returns than random. It does NOT
answer "is the detector finding the right days," because it never
looks at which days were truly part of an injected cluster.

This module does, because this project validates against synthetic
data with a known ground truth — something a real deployment never
has. label_ground_truth() turns generate_synthetic_data.py's
per-cluster ticker+date-range records into a per-(ticker, day) True/False
label; evaluate_detector() scores any of detect.py's three outputs
against that label with standard classification metrics.
"""

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# How many trading days before a cluster's first buy — through the date
# the forward price move begins — count as "truly" part of that
# informed-trading episode.
CLUSTER_LOOKBACK_DAYS = 5


def label_ground_truth(rows: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.Series:
    """
    True for a (ticker, date) row if it falls inside an injected
    cluster's window. ground_truth.csv only marks each cluster's ticker
    and date range; this expands that into a per-row label so standard
    classification metrics can be computed against it.
    """
    rows = rows.copy()
    rows["date"] = pd.to_datetime(rows["date"])
    labels = pd.Series(False, index=rows.index)
    for _, cluster in ground_truth.iterrows():
        start = pd.to_datetime(cluster.cluster_start) - pd.tseries.offsets.BDay(CLUSTER_LOOKBACK_DAYS)
        end = pd.to_datetime(cluster.jump_date) if pd.notna(cluster.jump_date) else pd.to_datetime(cluster.cluster_end)
        in_cluster = (rows["ticker"] == cluster.ticker) & (rows["date"] >= start) & (rows["date"] <= end)
        labels = labels | in_cluster
    return labels


def evaluate_detector(detections: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """
    Precision/recall/F1 at the detector's own operating point (its
    is_anomaly flag), plus threshold-independent PR-AUC/ROC-AUC from its
    continuous score. Rows with a NaN score (e.g. the unscored warm-up
    period of isolation_forest_detect_walkforward) are excluded — they
    were never given a chance to be flagged, so counting them as
    correctly-ignored negatives would flatter the detector.

    Returns None-valued metrics rather than a fabricated number if
    there's nothing to measure (no positive rows, or everything is one
    class after excluding unscored rows).
    """
    detections = detections[detections["score"].notna()]
    if len(detections) == 0:
        return {"precision": None, "recall": None, "f1": None,
                "pr_auc": None, "roc_auc": None, "n_true_positive_rows": 0, "n_scored_rows": 0}

    y_true = label_ground_truth(detections, ground_truth)
    n_positive = int(y_true.sum())

    if n_positive == 0 or n_positive == len(y_true):
        return {
            "precision": None, "recall": None, "f1": None,
            "pr_auc": None, "roc_auc": None,
            "n_true_positive_rows": n_positive, "n_scored_rows": len(detections),
        }

    y_pred = detections["is_anomaly"].values
    y_score = detections["score"].values

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "n_true_positive_rows": n_positive,
        "n_scored_rows": len(detections),
    }


def precision_recall_curve_points(detections: pd.DataFrame, ground_truth: pd.DataFrame):
    """(precision, recall) arrays across all score thresholds, for
    plotting — or None if there's no positive class to sweep against."""
    detections = detections[detections["score"].notna()]
    if len(detections) == 0:
        return None
    y_true = label_ground_truth(detections, ground_truth)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    precision, recall, _ = precision_recall_curve(y_true, detections["score"].values)
    return precision, recall
