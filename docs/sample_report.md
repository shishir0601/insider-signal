# Insider Trading Cluster Detector — Run Report

**Data**: synthetic (see README) — this run's ground truth injected 5 informed clusters across the dataset.

## Statistical baseline

- Flagged 26 of 2640 (ticker, day) rows (0.98%)
- Mean 10-day forward return, flagged days: **13.63%**
- Mean 10-day forward return, random control days: 0.66%
- t-test: t=7.60, p=0.0000 — statistically significant (p < 0.05)

## Isolation Forest

- Flagged 80 of 2640 (ticker, day) rows (3.03%)
- Mean 10-day forward return, flagged days: **7.70%**
- Mean 10-day forward return, random control days: 0.26%
- t-test: t=6.65, p=0.0000 — statistically significant (p < 0.05)

## Honest caveats

- This run is on synthetic data with a known injected signal, used to validate the pipeline finds what it should. Results on real SEC data would need real validation and are not implied by this report.
- No transaction costs, slippage, or multiple-comparison correction across tickers/days.
- Contamination rate for Isolation Forest was chosen manually, not cross-validated.