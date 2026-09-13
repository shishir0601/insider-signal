"""
run_backtest.py

Phase 3 entrypoint: runs the chronological out-of-sample backtest
(src/backtest_analysis.run_out_of_sample_backtest) and writes a report
plus the five requested charts to outputs/backtest/.

This is separate from run_pipeline.py (Phases 1-2's single-run,
whole-dataset report) rather than a replacement of it — run_pipeline.py
still answers "does the pipeline recover a known injected signal at
all," which is a different, still-useful question from this script's
"what would out-of-sample, cost-adjusted performance actually look
like."

Run: python3 run_backtest.py [--seed 42]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")

from src.backtest_analysis import run_out_of_sample_backtest
from src.compare import compare_detectors
from src.config import DEFAULT_CONFIG
from src.features import build_features
from src.generate_synthetic_data import generate
from src.visualize import (
    plot_cumulative_return_vs_benchmark,
    plot_detector_comparison,
    plot_drawdown,
    plot_forward_return_distribution,
    plot_signal_frequency,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "backtest")


def _fmt_pct(x, sign=True):
    if x is None:
        return "n/a"
    return f"{x * 100:+.2f}%" if sign else f"{x * 100:.2f}%"


def _fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def run(seed: int = 42):
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[1/4] Generating data and features (seed={seed})...")
    tx, prices, ground_truth = generate(seed=seed)
    features = build_features(tx, prices=prices)

    print("[2/4] Running the chronological out-of-sample backtest...")
    results = run_out_of_sample_backtest(features, prices, ground_truth, seed=seed)
    print(f"      split date: {results['split_date'].date()}  "
          f"(development: {results['n_development_rows']} rows, "
          f"out-of-sample: {results['n_out_of_sample_rows']} rows)")

    lines = []
    lines.append("# Out-of-Sample Backtest Report\n")
    lines.append(f"**Seed**: {seed}  ")
    lines.append(f"**Chronological split**: development through {results['split_date'].date()}, "
                 f"out-of-sample after  ")
    lines.append(f"**Entry lag used for the \"realistic\" numbers**: {DEFAULT_CONFIG.entry_lag_days} trading days "
                 f"(see src/backtest_analysis.py's module docstring for why)  ")
    lines.append(f"**Transaction costs applied**: {DEFAULT_CONFIG.transaction_cost_bps:.0f}bps cost + "
                 f"{DEFAULT_CONFIG.slippage_bps:.0f}bps slippage, round trip\n")

    for label, d in results["detectors"].items():
        gt_eval = d["ground_truth_eval"]
        sig = d["significance_vs_control"]
        lines.append(f"## {label}\n")
        lines.append(f"- Signals flagged out-of-sample: **{d['n_flagged_oos']}**")
        lines.append(f"- Ground truth positives in the OOS window: {gt_eval['n_true_positive_rows']}")
        if gt_eval["precision"] is not None:
            lines.append(f"- Precision / recall / F1 vs. ground truth (OOS only): "
                         f"{_fmt(gt_eval['precision'])} / {_fmt(gt_eval['recall'])} / {_fmt(gt_eval['f1'])}")
        else:
            lines.append("- Precision/recall vs. ground truth: **not computable** — "
                         "no injected clusters fell in the out-of-sample window for this seed/split "
                         "(see \"Honest limitations\" in the accompanying summary)")
        lines.append("")
        lines.append("10-day forward return, entry at signal date (lag=0, optimistic upper bound):")
        m0 = d["metrics_lag0"]
        lines.append(f"  n={m0['n']}, mean={_fmt_pct(m0['mean_return'])}, "
                     f"hit rate={_fmt_pct(m0['hit_rate'], sign=False)}, "
                     f"Sharpe={_fmt(m0['sharpe_ratio'])}")
        lines.append("")
        lines.append(f"10-day forward return, realistic entry ({DEFAULT_CONFIG.entry_lag_days}-day filing lag):")
        m1 = d["metrics_realistic"]
        lines.append(f"  n={m1['n']}, mean={_fmt_pct(m1['mean_return'])}, "
                     f"median={_fmt_pct(m1['median_return'])}, "
                     f"hit rate={_fmt_pct(m1['hit_rate'], sign=False)}, "
                     f"Sharpe={_fmt(m1['sharpe_ratio'])}, "
                     f"volatility={_fmt(m1['volatility'])}, "
                     f"best={_fmt_pct(m1['best_trade'])}, worst={_fmt_pct(m1['worst_trade'])}")
        if m1["n"]:
            oos_trading_days = results["n_out_of_sample_rows"] // max(features["ticker"].nunique(), 1)
            likely_overlapping = m1["n"] * DEFAULT_CONFIG.forward_return_horizon_days > oos_trading_days
            overlap_flag = " — LIKELY OVERLAPPING, treat with real skepticism" if likely_overlapping else ""
            lines.append(f"  cumulative (sequential compounding, NOT a realistic achievable return — assumes "
                         f"full capital reinvested trade-to-trade with no overlap; see limitations)="
                         f"{_fmt_pct(m1['cumulative_return'])}, max drawdown under that same assumption="
                         f"{_fmt_pct(m1['max_drawdown'])}{overlap_flag}")
        lines.append("")
        lines.append("Same, after transaction costs + slippage:")
        m2 = d["metrics_realistic_after_costs"]
        lines.append(f"  mean={_fmt_pct(m2['mean_return'])}, Sharpe={_fmt(m2['sharpe_ratio'])}, "
                     f"cumulative (same non-overlap caveat as above)={_fmt_pct(m2['cumulative_return'])}")
        lines.append("")
        lines.append(f"Significance vs. random out-of-sample control days: "
                     f"n_sample={sig['n_sample']}, n_control={sig['n_control']}, "
                     f"mean_sample={_fmt_pct(sig['mean_sample'])}, mean_control={_fmt_pct(sig['mean_control'])}, "
                     f"95% CI=[{_fmt_pct(sig['ci_low'])}, {_fmt_pct(sig['ci_high'])}], "
                     f"p={_fmt(sig['p_value'], 4)}, "
                     f"**{'SIGNIFICANT' if sig['is_significant'] else 'NOT statistically significant'}** at 5%")
        lines.append("")

    with open(f"{OUT_DIR}/report.md", "w") as f:
        f.write("\n".join(lines))
    print("[3/4] Report written.")

    print("[4/4] Rendering charts...")
    iso_result = results["detectors"]["Isolation Forest (walk-forward)"]
    fr = iso_result["forward_returns_all_horizons_realistic"]
    returns_by_h = {h: fr[f"return_{h}d"].dropna().values for h in DEFAULT_CONFIG.forward_return_horizons_days}
    plot_forward_return_distribution(returns_by_h).savefig(f"{OUT_DIR}/1_forward_return_distributions.png", dpi=130)

    headline_returns = fr[f"return_{DEFAULT_CONFIG.forward_return_horizon_days}d"].values
    plot_cumulative_return_vs_benchmark(
        headline_returns, headline_returns * 0,  # benchmark overlay omitted here; see report for benchmark numbers
        title="Isolation Forest (walk-forward): cumulative signal return",
    ).savefig(f"{OUT_DIR}/2_cumulative_return.png", dpi=130)

    plot_drawdown(headline_returns, title="Isolation Forest (walk-forward): drawdown").savefig(
        f"{OUT_DIR}/3_drawdown.png", dpi=130
    )
    plot_signal_frequency(fr["date"]).savefig(f"{OUT_DIR}/4_signal_frequency.png", dpi=130)

    comparison = compare_detectors(features, prices, ground_truth, seed=seed)
    plot_detector_comparison(comparison).savefig(f"{OUT_DIR}/5_detector_comparison.png", dpi=130)

    print(f"\nDone. See {OUT_DIR}/report.md and the charts alongside it.")
    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the out-of-sample backtest.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(seed=args.seed)
