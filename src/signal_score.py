"""
signal_score.py

A 0-100 score built from features.py's columns by an explicit,
hand-specified formula — not a fitted model, and not the same thing as
detect.py's detectors. The point of this module is interpretability: a
reader should be able to look at one row's score and see exactly which
factors pushed it up or down, without needing to trust a classifier's
internals. This is a presentation layer on top of the features, not a
fourth detector — src/compare.py and src/evaluation.py evaluate the
three actual detectors in detect.py, not this score.

WHY A HAND-SPECIFIED FORMULA, NOT A FITTED ONE: fitting weights (e.g.
logistic regression coefficients) against the synthetic ground truth
would tune this score to the specifics of one synthetic generator's
injection logic — the exact opposite of what a score meant to be read
by a person, and to generalize to real filings, should do. The weights
below are a deliberate, stated judgment call (documented next to each
one), not a result to be reported.

The five components, each contributing up to its own point cap (caps
sum to 100):

  - dollar_volume_zscore       up to 35 pts — is buying abnormal for this company?
  - distinct_buyers_5d         up to 25 pts — how many independent people?
  - role_weighted_buy_value_5d up to 15 pts — how senior are the buyers?
  - purchases_per_buyer_5d     up to 15 pts — how concentrated/repeated?
  - buy_sell_ratio_5d          up to 10 pts — how one-sided is it?

dollar_volume_zscore and distinct_buyers_5d carry the most weight
because they're exactly the two conditions statistical_baseline()
requires — this score is meant to read as "how close is this to
clearing the interpretable detector's bar, and by how much on each
axis," not an unrelated opinion.
"""

import pandas as pd

# (feature column, point cap, "feature value that earns full credit")
# Each component scales linearly from 0 at feature value 0 up to its
# point cap at the saturation value, then stays capped — e.g. a z-score
# of 5 or 8 both max out the 35-point dollar-volume component, because
# beyond "clearly abnormal" this score isn't trying to distinguish
# "very abnormal" from "extremely abnormal."
SCORE_COMPONENTS = [
    ("dollar_volume_zscore", 35, 5.0),
    ("distinct_buyers_5d", 25, 6.0),
    ("role_weighted_buy_value_5d", 15, 2_000_000.0),
    ("purchases_per_buyer_5d", 15, 3.0),
    ("buy_sell_ratio_5d", 10, 1.0),
]

COMPONENT_LABELS = {
    "dollar_volume_zscore": "Abnormal $ volume vs. this company's own history",
    "distinct_buyers_5d": "Number of independent insiders buying",
    "role_weighted_buy_value_5d": "Seniority of the buyers (CEO/CFO weighted higher)",
    "purchases_per_buyer_5d": "Repeated/concentrated buying",
    "buy_sell_ratio_5d": "How one-sided buying vs. selling is",
}


def _component_points(value, cap: float, saturation_value: float) -> float:
    """Points earned by one component, rounded to 1 decimal — shared by
    compute_signal_score() and explain_signal_score() so the displayed
    total and its breakdown can never silently disagree due to rounding
    at different granularities."""
    value = 0.0 if value is None or pd.isna(value) else max(float(value), 0.0)
    return round(min(value / saturation_value, 1.0) * cap, 1)


def compute_signal_score(features: pd.DataFrame) -> pd.Series:
    """Returns a 0-100 Series aligned to `features`'s index. A missing
    component column (e.g. hand-built test data with only some
    features) contributes 0, not an error."""
    score = pd.Series(0.0, index=features.index)
    for column, cap, saturation_value in SCORE_COMPONENTS:
        if column not in features.columns:
            continue
        component = features[column].apply(lambda v: _component_points(v, cap, saturation_value))
        score = score + component
    return score.clip(lower=0, upper=100).round(1)


def explain_signal_score(feature_row) -> list:
    """
    Per-component breakdown for one row (a pandas Series, e.g.
    features.iloc[i]) — what a report would show under "why did this
    score X": [(label, points_earned, points_cap), ...] in
    SCORE_COMPONENTS order. Sums exactly to compute_signal_score()'s
    value for the same row (both round at the same, per-component
    granularity — see _component_points).
    """
    breakdown = []
    for column, cap, saturation_value in SCORE_COMPONENTS:
        value = feature_row[column] if column in feature_row else None
        breakdown.append((COMPONENT_LABELS[column], _component_points(value, cap, saturation_value), cap))
    return breakdown
