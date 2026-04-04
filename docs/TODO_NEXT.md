# TODO Next

## Phase 1 — Input sanity check
Goal: quickly verify that the input dataset is internally valid before debugging OOS results.

Tasks:
- inspect columns and exact column names in `data/final_dataset.csv`
- confirm number of rows, date range, duplicate timestamps, missing values
- confirm return scale is correct
- check for impossible values or accidental cumulative series
- run basic descriptive statistics
- confirm weekly structure is acceptable

Deliverable:
- a short validation note
- exported summary tables / plots if useful

## Phase 2 — OOS returns debugging
Goal: verify that `results/oos_returns.csv` contains correct out-of-sample portfolio log returns.

Tasks:
- inspect all strategy columns
- check whether every column contains realistic positive and negative weeks
- identify any suspicious column with all-positive returns or zero drawdown
- recompute summary metrics directly from `results/oos_returns.csv`
- compare recomputed metrics with existing `metrics*.csv`
- trace the full generation pipeline from input data -> rolling window -> weights -> next-period OOS return
- verify no confusion between:
  - log returns vs simple returns
  - one-period returns vs cumulative wealth
  - in-sample vs out-of-sample slices
  - wrong indexing at rebalance dates
- if needed, rerun the generating script and rebuild `results/oos_returns.csv`

Deliverable:
- verified or corrected `results/oos_returns.csv`
- short bug report
- updated metrics computed from the corrected OOS file

## Phase 3 — Full data diagnostics on `data/final_dataset.csv`
Tasks:
- descriptive stats
- skewness
- kurtosis
- Jarque-Bera
- ADF
- KPSS
- Ljung-Box / autocorrelation
- ARCH effects
- rolling volatility
- rolling correlations
- crisis vs non-crisis descriptive comparison
- export results to csv/xlsx

## Phase 4 — Inference on `results/oos_returns.csv`
Tasks:
- Ledoit-Wolf Sharpe difference tests
- block bootstrap confidence intervals
- p-values for performance differences
- crisis vs non-crisis comparison
- optional turnover and net-of-cost comparison if retained

## Phase 5 — Structural metrics
Tasks:
- diversification ratio
- effective number of bets
- marginal risk contributions
- marginal CVaR contributions

Important:
These must be computed from portfolio weights and rolling risk objects, not only from final OOS return columns.

## Phase 6 — Thesis text alignment
Tasks:
- remove CFA from empirical model
- clarify RUONIA as short defensive / money market proxy
- remove claims not supported by implemented results
- eliminate repeated paragraphs
- fix methods and results sections to match actual computation

## Phase 7 — Final formatting
Tasks:
- formulas
- figures
- tables
- references
- table of contents
- headings
- numbering