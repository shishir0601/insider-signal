"""
run_pipeline.py

End-to-end run: generate data -> engineer features -> detect anomalies
(both the statistical baseline and Isolation Forest) -> backtest against
forward returns -> save charts and a summary report to outputs/.

Run: python3 run_pipeline.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.generate_synthetic_data import generate
from src.features import build_features
from src.detect import statistical_baseline, isolation_forest_detect
from src.backtest import backtest, forward_return

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("1/5  Generating data (synthetic — see README for why)...")
    tx, prices, ground_truth = generate()
    tx.to_csv("data/transactions.csv", index=False)
    prices.to_csv("data/prices.csv", index=False)
    ground_truth.to_csv("data/ground_truth.csv", index=False)

    print("2/5  Engineering features...")
    features = build_features(tx)

    print("3/5  Running detectors...")
    baseline = statistical_baseline(features)
    iso = isolation_forest_detect(features)

    print("4/5  Backtesting against forward returns...")
    baseline_result = backtest(baseline, prices, horizon_days=10)
    iso_result = backtest(iso, prices, horizon_days=10)

    print("5/5  Writing report and charts...")
    write_report(baseline, iso, baseline_result, iso_result, ground_truth)
    plot_returns(baseline_result, iso_result)
    plot_example_cluster(tx, prices, ground_truth, baseline)

    print(f"\nDone. See {OUT_DIR}/report.md and the charts alongside it.")


def write_report(baseline, iso, baseline_result, iso_result, ground_truth):
    lines = []
    lines.append("# Insider Trading Cluster Detector — Run Report\n")
    lines.append("**Data**: synthetic (see README) — this run's ground truth injected "
                  f"{len(ground_truth)} informed clusters across the dataset.\n")

    for name, dets, result in [("Statistical baseline", baseline, baseline_result), ("Isolation Forest", iso, iso_result)]:
        lines.append(f"## {name}\n")
        lines.append(f"- Flagged {int(dets.is_anomaly.sum())} of {len(dets)} (ticker, day) rows "
                      f"({dets.is_anomaly.mean()*100:.2f}%)")
        if result["flagged_mean_return"] is not None:
            lines.append(f"- Mean 10-day forward return, flagged days: **{result['flagged_mean_return']*100:.2f}%**")
            lines.append(f"- Mean 10-day forward return, random control days: {result['control_mean_return']*100:.2f}%")
        if result["p_value"] is not None:
            sig = "statistically significant (p < 0.05)" if result["p_value"] < 0.05 else "not statistically significant at p < 0.05"
            lines.append(f"- t-test: t={result['t_stat']:.2f}, p={result['p_value']:.4f} — {sig}")
        lines.append("")

    lines.append("## Honest caveats\n")
    lines.append("- This run is on synthetic data with a known injected signal, used to validate "
                  "the pipeline finds what it should. Results on real SEC data would need real "
                  "validation and are not implied by this report.")
    lines.append("- No transaction costs, slippage, or multiple-comparison correction across tickers/days.")
    lines.append("- Contamination rate for Isolation Forest was chosen manually, not cross-validated.")

    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines))


def plot_returns(baseline_result, iso_result):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, result) in zip(axes, [("Statistical baseline", baseline_result), ("Isolation Forest", iso_result)]):
        if len(result["flagged_returns"]) == 0:
            ax.set_title(f"{name} — no flags")
            continue
        ax.hist(result["control_returns"] * 100, bins=20, alpha=0.5, label="Control (random days)", color="#8a8580")
        ax.hist(result["flagged_returns"] * 100, bins=20, alpha=0.7, label="Flagged days", color="#a3792f")
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("10-day forward return (%)")
        ax.set_ylabel("count")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle("Forward returns: flagged vs. random control days")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "forward_returns.png"), dpi=130)
    plt.close(fig)


def plot_example_cluster(tx, prices, ground_truth, baseline_detections):
    if len(ground_truth) == 0:
        return
    example = ground_truth.iloc[0]
    ticker = example["ticker"]
    px = prices[prices.ticker == ticker].sort_values("date")
    px["date"] = pd.to_datetime(px["date"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(px["date"], px["close"], color="#232120", linewidth=1.4)

    cluster_start = pd.to_datetime(example["cluster_start"])
    cluster_end = pd.to_datetime(example["cluster_end"])
    ax.axvspan(cluster_start, cluster_end, color="#a3792f", alpha=0.25, label="Injected insider cluster window")

    flagged_dates = baseline_detections[(baseline_detections.ticker == ticker) & baseline_detections.is_anomaly]["date"]
    for d in pd.to_datetime(flagged_dates):
        ax.axvline(d, color="#96382e", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_title(f"{ticker}: price path, injected cluster window, and detector flags")
    ax.set_ylabel("price ($)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "example_cluster.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
