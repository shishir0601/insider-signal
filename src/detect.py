"""
detect.py

Three ways to flag a (ticker, day) row, deliberately kept separate so
they can be compared on equal footing (see src/compare.py):

1. Random baseline (random_baseline_detect).
   Not a strawman — the actual floor. If the other two detectors can't
   beat flagging days at random at the same rate, on the same ground
   truth, they aren't finding anything real.

2. Statistical baseline (statistical_baseline).
   A z-score threshold + minimum distinct buyers. Fully interpretable —
   you can explain to anyone exactly why a day got flagged, using only
   information available on or before that day (see the function
   docstring for exactly what "cluster" means and why each threshold
   exists). This is the floor any ML approach needs to beat.

3. Isolation Forest (isolation_forest_detect / isolation_forest_detect_walkforward).
   Unsupervised — chosen because there's no reliable label for "this
   was real insider trading" in a real deployment (see the function
   docstring for why Isolation Forest specifically, and why not a
   supervised model). Not called "AI" anywhere in this project: it's a
   specific, well-understood algorithm (random partitioning; points
   that isolate in fewer splits are more anomalous), and naming it
   plainly is more useful to a reader than a marketing label would be.

All three return the same shape (ticker, date, is_anomaly, score) so
they're directly comparable — including on the days already known to
be anomalies from the synthetic ground truth in generate_synthetic_data.py.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import DEFAULT_CONFIG
from src.features import NUMERIC_FEATURE_COLUMNS

# Isolation Forest trains on every numeric feature features.py produces
# (see that module for what each one means) — not a hand-picked subset.
# Keeping this as an explicit, named list (rather than "whatever columns
# happen to be in the DataFrame") means adding an unrelated column to
# `features` later can't silently change what the model trains on.
FEATURE_COLUMNS = list(NUMERIC_FEATURE_COLUMNS)


def random_baseline_detect(
    features: pd.DataFrame,
    flag_rate: float = DEFAULT_CONFIG.isolation_forest_contamination,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Flags a `flag_rate` fraction of (ticker, day) rows uniformly at
    random (default matches Isolation Forest's contamination rate, so
    the three detectors flag comparable NUMBERS of rows and any
    difference in precision/recall is about which rows, not how many).
    """
    out = features[["ticker", "date"]].copy()
    if len(out) == 0:
        out["score"] = pd.Series(dtype=float)
        out["is_anomaly"] = pd.Series(dtype=bool)
        return out

    rng = np.random.default_rng(seed)
    out["score"] = rng.uniform(0.0, 1.0, size=len(out))
    if flag_rate <= 0:
        out["is_anomaly"] = False
    elif flag_rate >= 1:
        out["is_anomaly"] = True
    else:
        threshold = np.quantile(out["score"], 1 - flag_rate)
        out["is_anomaly"] = out["score"] >= threshold
    return out[["ticker", "date", "is_anomaly", "score"]]


def statistical_baseline(
    features: pd.DataFrame,
    zscore_threshold: float = DEFAULT_CONFIG.zscore_threshold,
    min_distinct_buyers: int = DEFAULT_CONFIG.min_distinct_buyers,
) -> pd.DataFrame:
    """
    WHAT COUNTS AS A "CLUSTER": at least `min_distinct_buyers` (default
    3) different insiders each buying at some point within the trailing
    `trailing_window_days` (default 5) window ending on this day — see
    features.py's distinct_buyers_5d. One person buying, however large
    the trade, is not a cluster by this definition; three people buying
    independently within the same 5-day span is, regardless of size.

    WHAT ELSE HAS TO BE TRUE: the window's total buy $ volume also has
    to be a statistical outlier — `dollar_volume_zscore >= zscore_threshold`
    (default 2.5) — against THAT TICKER'S OWN trailing 60-day history
    (features.py's baseline_window_days). A day is flagged only if BOTH
    conditions hold.

    WHY BOTH CONDITIONS, NOT ONE: distinct-buyer count alone would flag
    a normal week for a company where several insiders routinely trade
    small amounts (e.g. a scheduled 10b5-1 plan touching multiple
    people). Dollar volume alone would flag one insider's single large,
    routine trade (a large stock award, a planned diversification sale
    that happens to look big for this company). Requiring both is what
    separates "several different people independently did something
    unusual at the same time" — hard to explain away — from either
    "one big trade" or "several small ones," either of which has an
    innocent, low-information explanation.

    WHY z >= 2.5 SPECIFICALLY: under a normal distribution, less than
    ~0.6% of days would clear 2.5 standard deviations by chance alone —
    a deliberately conservative bar so the interpretable detector stays
    genuinely conservative (see the precision/recall discussion in
    src/evaluation.py and the README: this detector trades recall for a
    low false-positive rate by design, not by accident).

    WHY min_distinct_buyers = 3: two people can plausibly coordinate
    unremarkably (a founder and their spouse, both directors); three or
    more independent insiders moving together is harder to attribute to
    coincidence or a single shared, mundane reason.

    WHAT'S AVAILABLE AT SIGNAL TIME / NO LOOK-AHEAD: both
    dollar_volume_zscore and distinct_buyers_5d are built from
    `.rolling()` windows in features.py, which only look backward from
    each date — a flag on day d never uses a transaction dated after d.
    See test/test_features_v2.py::TestNoLeakage.
    """
    out = features.copy()
    out["is_anomaly"] = (
        (out["dollar_volume_zscore"] >= zscore_threshold) & (out["distinct_buyers_5d"] >= min_distinct_buyers)
    )
    out["score"] = out["dollar_volume_zscore"]
    return out[["ticker", "date", "is_anomaly", "score"]]


def isolation_forest_detect(
    features: pd.DataFrame,
    contamination: float = DEFAULT_CONFIG.isolation_forest_contamination,
    seed: int = 42,
) -> pd.DataFrame:
    """
    WHY ISOLATION FOREST: it works by repeatedly picking a random
    feature and a random split point; a point that separates from the
    rest of the data in fewer such splits is more anomalous (isolating
    an outlier takes fewer, less-lucky cuts than isolating a point deep
    in a dense cluster of normal days). That's a good fit here because
    there's no reliable LABEL for "this was real insider trading" — in
    a real deployment there's no confirmed-fraud dataset to train a
    supervised classifier against, only the (synthetic, for validation
    only) ground truth this project happens to have. Anomaly detection
    is the right frame when the positive class is rare and unconfirmed;
    a supervised model would need labels this problem doesn't have.

    WHICH FEATURES: every column in FEATURE_COLUMNS (== every numeric
    feature features.py produces — see that module for what each one
    means), fillna(0)'d. Not "AI" in any load-bearing sense here: it's
    this specific, well-understood partitioning algorithm on this
    specific, named feature list.

    HOW `contamination` IS CHOSEN: it's an assumed anomaly RATE (default
    3%), not fit from data — scikit-learn uses it to pick the
    score threshold that flags roughly that fraction of rows. It is a
    genuine assumption, stated as one; see src/evaluation.py for how
    well that assumption's resulting flags actually hold up against
    ground truth, and README "Honest limitations" for why it isn't
    cross-validated (there's no real-data ground truth to validate it
    against yet).

    HOW ANOMALIES BECOME SIGNALS: `score` is
    -model.decision_function(X) (flipped so higher = more anomalous,
    which reads more intuitively than sklearn's own higher-is-more-normal
    convention); `is_anomaly` is whether sklearn's own predict() called
    it an outlier at the given contamination rate.

    LEAKAGE NOTE: this fits ONE model on every row at once, including,
    for any given day, rows from LATER days — that's look-ahead leakage
    if the question is "would this have flagged the day in real time."
    Fine for a single illustrative report; use
    isolation_forest_detect_walkforward() below for a leakage-safe
    evaluation, which is what src/compare.py uses.
    """
    out = features.copy()
    X = out[FEATURE_COLUMNS].fillna(0).values
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    raw_scores = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    # decision_function: higher = more normal, so flip sign for an
    # "anomalousness score" where higher = more anomalous (more intuitive).
    out["score"] = -model.decision_function(X)
    out["is_anomaly"] = raw_scores == -1
    return out[["ticker", "date", "is_anomaly", "score"]]


def isolation_forest_detect_walkforward(
    features: pd.DataFrame,
    contamination: float = DEFAULT_CONFIG.isolation_forest_contamination,
    seed: int = 42,
    min_train_rows: int = DEFAULT_CONFIG.walkforward_min_train_rows,
    refit_frequency_days: int = DEFAULT_CONFIG.walkforward_refit_frequency_days,
) -> pd.DataFrame:
    """
    The leakage-safe version of isolation_forest_detect(): refits a
    model every `refit_frequency_days` trading days using only rows
    dated strictly BEFORE that refit point, then scores that block of
    days with it before refitting on the (now larger) history and
    moving to the next block. A day is never scored by a model that has
    seen anything dated on or after it.

    Days before `min_train_rows` of history exists are left unscored
    (is_anomaly=False, score=NaN) rather than fit on too little data to
    mean anything — expect a NaN-scored stretch at the start of any run.
    """
    out = features.copy().sort_values("date").reset_index(drop=True)
    n = len(out)
    scores = np.full(n, np.nan)
    is_anomaly = np.zeros(n, dtype=bool)

    if n > 0:
        unique_dates = sorted(out["date"].unique())
        X_all = out[FEATURE_COLUMNS].fillna(0).values

        i = 0
        while i < len(unique_dates):
            block_dates = unique_dates[i: i + refit_frequency_days]
            train_mask = (out["date"] < block_dates[0]).values
            score_mask = out["date"].isin(block_dates).values

            if train_mask.sum() >= min_train_rows:
                model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
                model.fit(X_all[train_mask])
                X_block = X_all[score_mask]
                scores[score_mask] = -model.decision_function(X_block)
                is_anomaly[score_mask] = model.predict(X_block) == -1

            i += refit_frequency_days

    out["score"] = scores
    out["is_anomaly"] = is_anomaly
    return out[["ticker", "date", "is_anomaly", "score"]]
