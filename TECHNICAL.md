# Technical Documentation

This explains how InsiderSignal works and why specific decisions were made. It's for understanding the codebase, not for marketing it — see `README.md` for the project overview and results.

## System architecture

```
src/generate_synthetic_data.py   synthetic data generator, known ground truth
src/edgar/                       real SEC EDGAR ingestion
  client.py                        HTTP client (ticker->CIK, filing list, filing XML)
  parser.py                        Form 4 XML -> raw per-transaction dicts
  normalize.py                     raw dicts -> shared transaction schema
  ingest.py                        ties client+parser+normalize together
src/cleaning.py                  validates/normalizes transactions from either source
src/config.py                    every tunable parameter, in one place
src/features.py                  per-(ticker, day) feature engineering
src/detect.py                    three detectors: random, statistical, Isolation Forest
src/signal_score.py              interpretable 0-100 score (presentation layer, not a detector)
src/backtest.py                  single-horizon forward-return backtest + t-test
src/backtest_analysis.py         multi-horizon, benchmark, costs, temporal split, significance
src/evaluation.py                precision/recall/F1 against synthetic ground truth
src/compare.py                   runs all three detectors through both evaluation lenses
src/visualize.py                 matplotlib charts, reused by run_pipeline.py, run_backtest.py, and dashboard/
dashboard/                       Flask research-tool UI over the same src/ code
run_pipeline.py                  fast single-run illustrative report
run_backtest.py                  rigorous out-of-sample backtest report
run_dashboard.py                 interactive dashboard entrypoint
```

Design principle followed throughout: `src/` has no idea a dashboard or a report script exists. Both are thin layers that call the same functions; nothing is duplicated between them.

## SEC data pipeline

1. `EdgarClient.resolve_cik(ticker)` — ticker → 10-digit CIK via SEC's bulk `company_tickers.json`.
2. `EdgarClient.list_form4_filings(cik)` — reads `data.sec.gov/submissions/CIK##########.json`, filters to form types `4` and `4/A`. Only reads `filings.recent`; a filer with a long history has older filings paginated elsewhere, which this client doesn't follow (see Limitations).
3. `EdgarClient.fetch_filing_xml(...)` — fetches the actual `ownershipDocument` XML from `sec.gov/Archives/edgar/data/...`.
4. `parser.parse_form4_xml(xml)` — walks `<nonDerivativeTransaction>` elements (stdlib `xml.etree.ElementTree`, no new dependency). Deliberately ignores `<nonDerivativeHolding>` (a position with no transaction) and `<derivativeTransaction>` (options/RSUs — the corresponding "M" non-derivative line is what records the resulting common-stock transaction, which is what this schema tracks).
5. `normalize.normalize_edgar_records(records)` — maps into the shared schema (below), classifying `transaction_type` from the acquired/disposed code (A/D) and `is_open_market` from the transaction code (P/S = open market; M/A/F/G/... = mechanical, e.g. option exercises, tax withholding, grants — captured but not yet used by detection, see README's Future Work).

**Required `EDGAR_USER_AGENT`**: SEC rejects (403) unidentified requests. `EdgarClient` has no default — a shared placeholder would get everyone who forgot to change it blocked together. Rate-limited client-side to stay under SEC's ~10 req/sec.

### Normalized transaction schema

```
ticker, insider_id, date, transaction_type, shares, price     # required — features.py needs exactly these
insider_name, insider_title, transaction_code,                # enrichment — present for real data,
transaction_code_label, is_open_market, total_value,           # optional for synthetic (features.py
issuer_name                                                     # degrades gracefully if absent)
```

Both `generate_synthetic_data.generate()` and `edgar.normalize_edgar_records()` produce this schema, which is what lets identical code run on either source.

## Data cleaning

`src/cleaning.py` runs seven ordered steps (missing fields → invalid numerics → invalid dates → invalid transaction type → text normalization → minimum purchase value → duplicates), returning a `CleaningReport` that counts every dropped row by reason. Nothing is silently discarded. Order matters: normalization happens before duplicate detection so `"AAPL"` and `" aapl "` are recognized as the same entity instead of hiding a real duplicate.

Date validity is checked against a **fixed plausibility range** (1994–2100), not "not in the future relative to now" — a wall-clock-relative check would make cleaning behavior depend on when the code happens to run, which is wrong for a deterministic batch process (this was an actual bug caught during development: synthetic test data with dates in "2026" started failing the check once real time passed that point).

## Feature engineering

Computed per (ticker, day) over a trailing window (`trailing_window_days`, default 5 — this is what "cluster window" means throughout). See README for the feature list and their meaning. Two implementation notes:

- **`insider_relative_size_zscore_5d`** compares a trade only to that same insider's own prior trades (strictly earlier dates), never to a different insider's history or to the company's aggregate — computed transaction-by-transaction in date order, not as a rolling window over the whole series.
- **`role_weighted_buy_value_5d`** uses a substring-matching lookup (`config.role_weights`) rather than exact title matching, since real Form 4 titles are free text ("Chief Financial Officer" vs. "CFO" vs. "EVP and CFO").

## Cluster definition

A "cluster" (as used by the statistical baseline) requires **both**: at least `min_distinct_buyers` (default 3) different insiders buying within the trailing window, **and** that window's total buy $ volume clearing `zscore_threshold` (default 2.5) standard deviations above the ticker's own trailing 60-day history. Both conditions are required because either alone has an innocent explanation (a scheduled multi-person trading plan; one large routine trade) that clustering-plus-abnormal-volume together does not.

## Statistical detector

`src/detect.py:statistical_baseline()`. Fully interpretable by construction — every flag can be explained by the exact rule above using only the two named columns. Deliberately conservative (a 2.5-sigma bar means well under 1% of days clear it by chance alone under a normality assumption); trades recall for a low false-positive rate.

## Isolation Forest

Chosen because there's no reliable label for "this was real insider trading" in production — anomaly detection is the right frame when the positive class is rare and unconfirmed; a supervised model would need labels this problem doesn't have. Trained on every column in `features.NUMERIC_FEATURE_COLUMNS`. `contamination` (default 0.03) is an assumed anomaly rate, not fit from data — stated as an assumption, not validated against real-world ground truth (none exists yet).

Two variants exist on purpose:
- `isolation_forest_detect()` — fits once on the whole dataset. Fast, useful for a single illustrative run, but has look-ahead leakage: a day early in the dataset is scored by a model that has already seen every later day.
- `isolation_forest_detect_walkforward()` — refits every `walkforward_refit_frequency_days` (default 20) trading days, using only rows strictly before that point. This is what the backtest and dashboard use. Days before `walkforward_min_train_rows` (default 200) of history exists are left unscored (NaN), not fit on too little data to mean anything.

## Signal score

`src/signal_score.py`. A hand-specified, capped linear combination of five features (point caps sum to 100), not a fitted model — see README for why. Every component's points are rounded once, at the same granularity used for both the total and the per-factor breakdown, specifically so the two can never silently disagree due to rounding at different precision (an actual bug caught during development, where the displayed total and its explanation summed to different numbers).

## Backtesting methodology

`src/backtest_analysis.py`. Key decisions:

- **Entry lag**: the `date` column is the transaction date, not the filing date. Form 4 is legally due within 2 business days of the transaction. Backtesting entry at the transaction date (`lag=0`) assumes information no real trader had yet; every multi-horizon function accepts `entry_lag_days` and the reported "realistic" numbers use `config.entry_lag_days` (2).
- **Benchmark**: equal-weighted average forward return across every ticker in the universe over the same dates — stated as a universe-relative benchmark, not a real market index (none exists for synthetic tickers).
- **Cumulative return / drawdown**: computed by compounding trades in chronological order, which assumes non-overlapping, fully-reinvested capital. Insider-signal trades often DO overlap in real calendar time; wherever these numbers are reported, an explicit caveat (and an automatic overlap heuristic) accompanies them rather than presenting a compounded figure as an achievable return.
- **Transaction costs**: a flat round-trip deduction (`2 * (transaction_cost_bps + slippage_bps) / 10000`), not size- or liquidity-aware.

## Walk-forward methodology

`run_out_of_sample_backtest()`: `split_by_date()` picks a chronological cutoff (never a random shuffle, which would leak information across the split via nearby dates for the same ticker). Everything at or before the split is "development"; nothing is currently tuned there (all thresholds are fixed, documented constants), so the split today enforces discipline for the moment that changes, and gives "out-of-sample" real meaning now. All reported metrics are computed exclusively on rows dated after the split. The walk-forward Isolation Forest is still fit using full history up to each scored date (which naturally includes development-period rows — exactly how it would behave in production) but is never scored on a development-period date.

## Statistical tests

Welch's t-test (unequal variance assumed) between sample and control returns, via `scipy.stats.ttest_ind`. A t-distribution-based confidence interval for the sample mean (`scipy.stats.t.ppf`). Both honestly report `None`/`is_significant=False` when either sample has fewer than 2 observations, rather than computing a number that wouldn't mean anything.

## Look-ahead-bias prevention

Every feature for day *d* uses only transactions/prices dated on or before *d*:
- `features.py`'s rolling computations use pandas `.rolling()` (backward-looking by construction) and `.pct_change()` for the pre-signal return.
- `insider_relative_size_zscore_5d` is built transaction-by-transaction in date order, using only each insider's strictly-prior trades.
- The walk-forward Isolation Forest never trains on same-or-later data than it scores (enforced structurally, and checked by tests that assert this for every scored date).
- `compute_forward_returns()`'s entry-lag shift and `split_by_date()`'s chronological cutoff are both tested against synthetic perturbation: changing data strictly after an observation date must not change that date's computed values (`test/test_features_v2.py::TestNoLeakage`, `test/test_backtest_analysis.py::TestNoLeakageInBacktestAnalysis`).
- The one place look-ahead is intentionally present — `isolation_forest_detect()`'s whole-dataset fit — is documented as such everywhere it's used, and is never the version used for a reported backtest number.

## Important parameters

All in `src/config.py`, with the reasoning for each default documented inline: `trailing_window_days`, `baseline_window_days`, `zscore_threshold`, `min_distinct_buyers`, `isolation_forest_contamination`, `forward_return_horizon_days`, `min_purchase_value`, `insider_history_min_trades`, `pre_signal_return_window_days`, `role_weights`, `walkforward_min_train_rows`, `walkforward_refit_frequency_days`, `forward_return_horizons_days`, `entry_lag_days`, `transaction_cost_bps`, `slippage_bps`, `development_period_fraction`.

## Limitations

See README's Limitations section — kept in one place rather than duplicated here.
