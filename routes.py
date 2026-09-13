"""
routes.py

One page, full-reload on search (a plain HTML form, not a JS SPA) — a
small research tool doesn't need more than that. Chart rendering
(matplotlib -> base64 PNG, embedded inline) happens here, not in
data_service.py, to keep that module's output plain data that's easy
to test without touching matplotlib.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request

from dashboard.data_service import analyze_ticker
from src.backtest_analysis import compute_benchmark_returns, compute_forward_returns
from src.visualize import (
    plot_cumulative_return_vs_benchmark,
    plot_detector_comparison,
    plot_forward_return_distribution,
    plot_price_with_signals,
    plot_signal_history_returns,
)

bp = Blueprint("dashboard", __name__)


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


@bp.get("/")
def index():
    ticker = request.args.get("ticker", "").strip()
    result = None
    charts = {}

    if ticker:
        result = analyze_ticker(ticker)

        if result.get("found") and result.get("is_demo"):
            demo = result["_demo"]
            charts["price"] = _fig_to_data_uri(plot_price_with_signals(
                result["_ticker_prices"], result["_ticker_buys"], result["_ticker_signal_dates"],
                ticker=result["ticker"],
            ))

            detections = demo["combined_detections"][demo["combined_detections"].ticker == result["ticker"]]
            fr = compute_forward_returns(detections, demo["prices"], horizons_days=(1, 5, 10, 20))
            if len(fr):
                returns_by_h = {h: fr[f"return_{h}d"].dropna().values for h in (1, 5, 10, 20)}
                charts["forward_returns"] = _fig_to_data_uri(plot_forward_return_distribution(returns_by_h))
                charts["history"] = _fig_to_data_uri(
                    plot_signal_history_returns(fr["date"], fr["return_10d"].fillna(0).values)
                )

                bench = compute_benchmark_returns(demo["prices"], fr["date"], horizons_days=(10,))
                bench_10d = bench.set_index("date").reindex(fr["date"])["benchmark_return_10d"].fillna(0).values
                charts["cumulative"] = _fig_to_data_uri(plot_cumulative_return_vs_benchmark(
                    fr["return_10d"].fillna(0).values, bench_10d,
                    title=f"{result['ticker']}: cumulative signal return vs. benchmark",
                ))

            # Global (all-ticker) comparison, precomputed once and
            # cached with the demo dataset — see data_service.py for
            # why this is global rather than per-ticker.
            charts["comparison"] = _fig_to_data_uri(plot_detector_comparison(
                demo["detector_comparison"], metric_columns=["precision", "recall", "f1"]
            ))

    return render_template("index.html", ticker=ticker, result=result, charts=charts)
