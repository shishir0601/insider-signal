"""
config.py

Centralizes the parameters that were previously hardcoded inside
features.py, detect.py, and backtest.py, plus the new parameters the
real-data cleaning layer needs. Every function that reads one of these
still takes it as a normal keyword argument with the same default it
always had — this file is a single place to see and change those
defaults, not a new layer of indirection call sites have to go through.

A plain dataclass on purpose: this project has one deployment target
(a developer's machine), so a settings framework (pydantic-settings,
Hydra, etc.) would be solving a problem it doesn't have. Override a
value by editing the constant below, or by passing it explicitly to
the function that uses it.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineConfig:
    # --- features.py: the rolling "cluster window" and its baseline ---
    trailing_window_days: int = 5
    """How many trailing trading days count as one buying window — the
    "cluster window" insiders' purchases are grouped into."""

    baseline_window_days: int = 60
    """Trailing history a ticker's own $ buy volume is compared against
    to compute its z-score (the "lookback window")."""

    # --- detect.py: statistical baseline ---
    zscore_threshold: float = 2.5
    """Minimum dollar-volume z-score for a day to be flagged."""

    min_distinct_buyers: int = 3
    """Minimum number of different insiders required in the trailing
    window for a day to count as a "cluster" rather than one person's
    routine trade."""

    # --- detect.py: Isolation Forest ---
    isolation_forest_contamination: float = 0.03
    """Expected fraction of (ticker, day) rows that are anomalies —
    the anomaly threshold for the unsupervised detector."""

    # --- backtest.py ---
    forward_return_horizon_days: int = 10
    """How many trading days ahead a flagged day's forward return is
    measured over."""

    # --- cleaning.py: real-data-specific (synthetic data doesn't need these) ---
    min_purchase_value: float = 0.0
    """Transactions worth less than this are excluded before feature
    engineering (0 disables the filter). Real Form 4 data includes tiny
    transactions — a few dollars of rounding, dividend reinvestment —
    that aren't meaningful signals of informed buying; this is not
    needed for the synthetic generator, which doesn't produce them."""

    # --- features.py: Phase 2 additions ---
    insider_history_min_trades: int = 3
    """Minimum number of an insider's own PRIOR trades required before
    their "relative to own history" z-score is considered meaningful
    (see features.insider_relative_size_zscore). Below this, there
    isn't enough history to say a trade is unusually large for THEM —
    the feature falls back to a neutral 0.0 rather than a noisy
    estimate from one or two data points."""

    pre_signal_return_window_days: int = 10
    """How many trading days of price history before the signal date
    the "stock return before the signal" feature looks back over. Set
    to match forward_return_horizon_days by default so the "before" and
    "after" windows are directly comparable in the report."""

    role_weights: dict = field(default_factory=lambda: {
        "chief executive": 3.0, "ceo": 3.0,
        "chief financial": 3.0, "cfo": 3.0,
        "president": 2.5,
        "chief operating": 2.5, "coo": 2.5,
        "director": 2.0,
        "10% owner": 1.5,
        "officer": 1.5,
    })
    """Substring (case-insensitive) -> weight, used to weight a BUY
    transaction's dollar value by the insider's seniority when computing
    role_weighted_buy_value_5d. Checked in dict order; the first
    matching substring wins. A title matching none of these (or a
    missing title) gets weight 1.0 — see
    src/signal_score.py:ROLE_WEIGHT_DEFAULT. Chosen as a simple,
    inspectable lookup rather than a learned weighting: the point of
    this feature is that a CFO's purchase is prima facie more
    informative than an unspecified "Other" filer's, not a fitted
    coefficient that would need real labeled outcomes to justify."""

    # --- detect.py: walk-forward Isolation Forest ---
    walkforward_min_train_rows: int = 200
    """Minimum number of (ticker, day) rows required before the
    walk-forward Isolation Forest will fit a model at all. Below this,
    rows are left unscored (is_anomaly=False, score=NaN) rather than
    fit on too little data to mean anything."""

    walkforward_refit_frequency_days: int = 20
    """How often (in trading days) the walk-forward Isolation Forest
    refits. Smaller = more realistic (closer to refitting daily) but
    slower; larger = faster but the model goes longer without seeing
    recent data."""

    # --- backtest_analysis.py: Phase 3 additions ---
    forward_return_horizons_days: tuple = (1, 5, 10, 20)
    """The set of holding periods every signal's forward return is
    computed over. 10 stays the default single-horizon reported
    elsewhere (backtest.py, src/compare.py) for continuity with
    Phases 1-2; this is the fuller multi-horizon view."""

    entry_lag_days: int = 2
    """Trading days between the TRANSACTION date (what this project's
    `date` column records) and the earliest a real trader could have
    known about it. Form 4 filings are legally due within 2 business
    days of the transaction — often later in practice — so a signal
    dated `d` was not actually public knowledge until roughly `d +
    entry_lag_days`. Backtesting an entry at `d` itself (lag=0) is
    optimistic; see backtest_analysis.py and the README's look-ahead
    audit for why both are reported."""

    transaction_cost_bps: float = 10.0
    """One-way transaction cost in basis points (10 bps = 0.10%),
    applied on both entry and exit (round trip)."""

    slippage_bps: float = 5.0
    """One-way slippage in basis points, applied the same way as
    transaction_cost_bps. Kept separate so the two can be reasoned
    about (and reported) independently even though they're combined
    when adjusting a return."""

    development_period_fraction: float = 0.6
    """Fraction of the chronological date range treated as the
    "development" period for a temporal train/test split — see
    backtest_analysis.run_out_of_sample_backtest(). The remaining
    (1 - this) fraction, strictly later in time, is the out-of-sample
    period actually used for reported evaluation results."""


DEFAULT_CONFIG = PipelineConfig()
