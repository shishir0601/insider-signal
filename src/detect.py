"""
detect.py

Two detectors, deliberately kept separate:

1. Statistical baseline (z-score threshold + minimum distinct buyers).
   Fully interpretable — you can explain to anyone exactly why a day got
   flagged. This is the floor any ML approach needs to beat.

2. Isolation Forest (unsupervised, scikit-learn).
   Isolation Forest works by randomly partitioning the feature space and
   measuring how few splits it takes to isolate a point — anomalies get
   isolated in fewer splits because they sit far from the dense "normal"
   region. Chosen over a supervised classifier because we don't have (and
   in real deployment wouldn't have) reliable labels for "this actually
   was insider trading" — anomaly detection is the right frame for a
   problem where the positive class is rare and unconfirmed.

Both operate on the SAME feature table from features.py and return the
same shape of output (ticker, date, is_anomaly, score) so they're directly
comparable — including on the days we already know are anomalies from the
synthetic ground truth in generate_synthetic_data.py.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLUMNS = ["distinct_buyers_5d", "buy_dollar_volume_5d", "buy_sell_ratio_5d", "dollar_volume_zscore"]


def statistical_baseline(features: pd.DataFrame, zscore_threshold: float = 2.5, min_distinct_buyers: int = 3) -> pd.DataFrame:
    """
    A day is flagged if BOTH:
      - dollar volume is a statistical outlier vs. that ticker's own history, AND
      - at least `min_distinct_buyers` different insiders contributed.
    Requiring both conditions is what separates "one insider had a big
    trade" (routine) from "multiple insiders moved at once" (the signal
    that's actually hard to explain away).
    """
    out = features.copy()
    out["is_anomaly"] = (
        (out["dollar_volume_zscore"] >= zscore_threshold) & (out["distinct_buyers_5d"] >= min_distinct_buyers)
    )
    out["score"] = out["dollar_volume_zscore"]
    return out[["ticker", "date", "is_anomaly", "score"]]


def isolation_forest_detect(features: pd.DataFrame, contamination: float = 0.03, seed: int = 42) -> pd.DataFrame:
    out = features.copy()
    X = out[FEATURE_COLUMNS].fillna(0).values
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    raw_scores = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    # decision_function: higher = more normal, so flip sign for an
    # "anomalousness score" where higher = more anomalous (more intuitive).
    out["score"] = -model.decision_function(X)
    out["is_anomaly"] = raw_scores == -1
    return out[["ticker", "date", "is_anomaly", "score"]]
