"""
visualize.py

Five charts for the Phase 3 out-of-sample backtest report. Kept
deliberately plain matplotlib — readable and correct, not a design
exercise. Every function returns a Figure; callers decide whether to
save, show, or embed it.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_FLAGGED = "#3B6FA0"
COLOR_BENCHMARK = "#999999"
COLOR_NEGATIVE = "#B04A4A"


def plot_forward_return_distribution(returns_by_horizon: dict, benchmark_by_horizon: dict = None):
    """
    returns_by_horizon: {horizon_days: np.ndarray of returns}
    benchmark_by_horizon: same shape, optional — overlaid for comparison.
    One subplot per horizon, side by side.
    """
    horizons = sorted(returns_by_horizon.keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 4), squeeze=False)
    axes = axes[0]
    for ax, h in zip(axes, horizons):
        r = returns_by_horizon[h]
        r = r[~np.isnan(r)] if len(r) else r
        if len(r):
            ax.hist(r * 100, bins=min(20, max(5, len(r) // 2)), color=COLOR_FLAGGED, alpha=0.75, label="Signal")
        if benchmark_by_horizon and h in benchmark_by_horizon:
            b = benchmark_by_horizon[h]
            b = b[~np.isnan(b)] if len(b) else b
            if len(b):
                ax.hist(b * 100, bins=min(20, max(5, len(b) // 2)), color=COLOR_BENCHMARK, alpha=0.5, label="Benchmark")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{h}-day forward return")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.suptitle("Forward-return distributions by horizon")
    fig.tight_layout()
    return fig


def plot_cumulative_return_vs_benchmark(signal_returns: np.ndarray, benchmark_returns: np.ndarray, title: str = ""):
    """Compounds both return series in order and plots the two equity
    curves together. Both arrays should be the same length and
    chronologically ordered (matched signal-by-signal to a benchmark
    return over the same window)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sig = signal_returns[~np.isnan(signal_returns)]
    bench = benchmark_returns[~np.isnan(benchmark_returns)]

    if len(sig):
        sig_curve = np.cumprod(1 + sig)
        ax.plot(range(1, len(sig_curve) + 1), (sig_curve - 1) * 100, color=COLOR_FLAGGED, label="Signal (compounded)", linewidth=1.8)
    if len(bench):
        bench_curve = np.cumprod(1 + bench)
        ax.plot(range(1, len(bench_curve) + 1), (bench_curve - 1) * 100, color=COLOR_BENCHMARK, label="Benchmark (compounded)", linewidth=1.5, linestyle="--")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Signal number, chronological order — NOT calendar time; trades likely overlap in reality")
    ax.set_ylabel("Cumulative return (%)\n(sequential compounding, illustrative only)")
    ax.set_title(title or "Cumulative signal return vs. benchmark")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_drawdown(returns: np.ndarray, title: str = ""):
    """Drawdown of the same chronologically-compounded equity curve
    compute_performance_metrics() uses — see that function's docstring
    for the "trades taken back-to-back" assumption this shares."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    r = returns[~np.isnan(returns)]
    if len(r):
        equity = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        ax.fill_between(range(1, len(drawdown) + 1), drawdown, 0, color=COLOR_NEGATIVE, alpha=0.6)
        ax.plot(range(1, len(drawdown) + 1), drawdown, color=COLOR_NEGATIVE, linewidth=1)
    ax.set_xlabel("Signal number, chronological order — NOT calendar time; trades likely overlap in reality")
    ax.set_ylabel("Drawdown (%)\n(sequential compounding, illustrative only)")
    ax.set_title(title or "Drawdown")
    fig.tight_layout()
    return fig


def plot_signal_frequency(signal_dates: pd.Series, title: str = ""):
    """Signals per calendar month — how bursty vs. steady the detector's
    flagging behavior is over time."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    dates = pd.to_datetime(pd.Series(signal_dates))
    if len(dates):
        counts = dates.dt.to_period("M").value_counts().sort_index()
        ax.bar([str(p) for p in counts.index], counts.values, color=COLOR_FLAGGED)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Signals flagged")
    ax.set_title(title or "Signal frequency over time")
    fig.tight_layout()
    return fig


def plot_detector_comparison(comparison: pd.DataFrame, metric_columns: list = None):
    """
    comparison: a DataFrame indexed by detector name (e.g.
    src.compare.compare_detectors()'s output, or a small hand-built
    frame of Phase 3 metrics) with the requested numeric columns.
    """
    if metric_columns is None:
        metric_columns = [c for c in ["precision", "recall", "f1", "roc_auc"] if c in comparison.columns]
    fig, ax = plt.subplots(figsize=(1.6 * len(comparison) + 2, 4.5))
    x = np.arange(len(comparison.index))
    width = 0.8 / max(len(metric_columns), 1)
    for i, col in enumerate(metric_columns):
        values = comparison[col].astype(float).fillna(0).values
        ax.bar(x + i * width, values, width=width, label=col)
    ax.set_xticks(x + width * (len(metric_columns) - 1) / 2)
    ax.set_xticklabels(comparison.index, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Detector comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_price_with_signals(prices: pd.DataFrame, buy_transactions: pd.DataFrame = None,
                             signal_dates: pd.Series = None, ticker: str = ""):
    """
    Dashboard chart #1: a ticker's close price with insider BUY
    transactions marked (sized by $ value) and detector-flagged signal
    dates marked separately — the single chart the main dashboard view
    is built around ("stock price + insider purchases" + "signal
    markers" are one chart, not two, since a signal only means anything
    relative to the price it's plotted against).

    prices: this ticker's (date, close) rows.
    buy_transactions: this ticker's BUY rows (date, shares, price) — dot
    size scales with transaction $ value.
    signal_dates: dates any detector flagged for this ticker.
    """
    fig, ax = plt.subplots(figsize=(9, 4.2))
    prices = prices.sort_values("date")
    ax.plot(prices["date"], prices["close"], color="#2B2B2B", linewidth=1.3, zorder=2, label="Close price")

    if buy_transactions is not None and len(buy_transactions):
        dollar_value = buy_transactions["shares"] * buy_transactions["price"]
        sizes = 30 + 220 * (dollar_value / dollar_value.max()) if dollar_value.max() > 0 else 40
        ax.scatter(buy_transactions["date"], buy_transactions["price"], s=sizes, color="#3B6FA0",
                   alpha=0.65, edgecolors="none", zorder=3, label="Insider buy ($ = size)")

    if signal_dates is not None and len(signal_dates):
        for i, d in enumerate(pd.to_datetime(pd.Series(signal_dates)).unique()):
            ax.axvline(d, color="#B04A4A", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1,
                       label="Detected signal" if i == 0 else None)

    ax.set_title(f"{ticker}: price, insider buys, and detected signals" if ticker else "Price, insider buys, and detected signals")
    ax.set_ylabel("Close price ($)")
    ax.legend(fontsize=8, loc="upper left")
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return fig


def plot_signal_history_returns(signal_dates: pd.Series, returns: np.ndarray, title: str = ""):
    """
    Dashboard chart #3: how past signals actually performed over time —
    each historical signal's forward return plotted against its date
    (not just the distribution shape, which plot_forward_return_distribution
    already covers) — positive in the flagged color, negative in red,
    so a viewer can see whether performance is steady or concentrated
    in a few episodes.
    """
    fig, ax = plt.subplots(figsize=(8, 3.6))
    dates = pd.to_datetime(pd.Series(signal_dates))
    returns = np.asarray(returns, dtype=float)
    colors = [COLOR_FLAGGED if r >= 0 else COLOR_NEGATIVE for r in returns]
    if len(dates):
        ax.bar(dates, returns * 100, color=colors, width=3)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Forward return (%)")
    ax.set_title(title or "Historical signal performance over time")
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return fig
