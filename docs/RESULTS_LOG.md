# Results Log

## Purpose
This file stores verified intermediate results and decisions.
Only confirmed results should be written here.

---

## Entry template

### Date:
YYYY-MM-DD

### Task:
Short description of what was checked.

### Files used:
- file1.py
- file2.csv

### What was done:
- short bullet
- short bullet

### Key result:
A concise statement of the verified result.

### Action needed:
Next step based on this result.

---

## Entries

### Date:
2026-03-31

### Task:
Project reset and audit setup

### Files used:
- PROJECT_STATE.md
- CURRENT_ISSUES.md
- TODO_NEXT.md

### What was done:
- defined actual empirical scope
- documented main inconsistencies
- fixed immediate task order

### Key result:
The empirical thesis scope is crypto-extended Russian multi-asset portfolio without CFA in the implemented dataset.

### Action needed:
Run Phase 1 input sanity check and then debug `oos_returns.csv`.

---

### Date:
2026-03-31

### Task:
Phase 1/2 initial pipeline audit: validate `final_dataset.csv` and check `oos_returns.csv` sanity

### Files used:
- merge.py
- final_dataset.csv
- main.py
- riskparity.py
- Доходности обработка/allweather.py
- oos_returns.csv
- metrics.py

### What was done:
- traced the generating scripts: `merge.py` writes `final_dataset.csv`; `main.py` reads it and writes `oos_returns.csv`
- checked `final_dataset.csv` for shape, columns, date range, duplicates, and missing values
- checked `oos_returns.csv` for suspicious strategy columns (no negative weeks / near-zero volatility)
- compared `mv_sample_*` OOS series to `ruonia_return` to see if MV collapses into RUONIA

### Key result:
`final_dataset.csv` looks internally consistent (498 rows, 7 columns, no NaNs/duplicate dates; columns are BTC/ETH/Gold/RUONIA/MOEX/OFZ log returns), but its date range currently runs to 2026-03-15 (not 2025). In `oos_returns.csv`, `mv_sample_base` and `mv_sample_crypto` have **no negative weeks** and very low volatility; they are ~RUONIA (corr ≈ 0.998), meaning MV (long-only) effectively allocates almost entirely to RUONIA. Also, the backtests compute portfolio returns as a **dot product of asset log returns**, which is generally not the correct way to obtain portfolio log returns (should weight simple returns and then log), so `oos_returns.csv` is likely not “true” OOS portfolio log returns as labeled.

### Action needed:
Fix the OOS return construction: convert asset log returns to simple, compute portfolio simple return with weights, then convert to log; decide whether MV is allowed to invest in RUONIA (or treat RUONIA as rf / exclude / constrain).

---

### Date:
2026-03-31

### Task:
Fix OOS return construction pipeline and regenerate outputs

### Files used:
- riskparity.py
- rpallw.py
- allweather.py
- metrics.py
- main.py
- export_thesis_metrics.py
- final_dataset.csv
- oos_returns.csv (old + new)
- metrics*.csv (old + new)

### What was done:
- implemented correct portfolio return construction for log-return inputs: \(R_i = \exp(r_i)-1\), \(R_p=\sum w_i R_i\), \(r_p=\log(1+R_p)\)
- applied this across EW, MV, RP (in `riskparity.py`), Hybrid variants (in `rpallw.py`), and All-Weather (new canonical `allweather.py`)
- backed up old outputs with `.old` suffix and regenerated `oos_returns.csv`, `metrics.csv`, `metrics_full.csv`, `metrics_crisis.csv`, `metrics_non_crisis.csv`, `l1_distances.csv`
- updated `metrics.py` so performance metrics are computed consistently when OOS returns are stored as log returns (conversion to simple where needed)
- compared old vs new OOS returns and flagged remaining anomalies / duplicates

### Key result:
The issue was a **return-definition bug** (portfolio “log returns” were computed as \(w^T r^{log}\), which is not the correct portfolio log return). After the fix, most strategy series changed modestly (largest mean absolute per-week change ≈ 0.00134 for `ew_crypto`; correlations old vs new remain ~0.998+), indicating directionally similar but numerically incorrect prior results. `mv_sample_base` and `mv_sample_crypto` still have no negative weeks and zero drawdown, and remain ~RUONIA (corr ≈ 0.998, max abs diff ≈ 0.00029); this is a **legitimate consequence of model design** (long-only min-variance with RUONIA available), not an implementation bug. Hybrid variants `hybrid_*`, `hybrid_loose_*`, `hybrid_unconstrained_*` are **numerically identical** in current code because they call the same generic routine with the same parameters; this is an implementation/design issue in naming/parameterization (not caused by the return fix).

### Action needed:
- Decide whether to keep reporting MV-sample as “min variance” (documenting that it becomes RUONIA-heavy) or to present an additional MV variant excluding RUONIA as a separate model (only if explicitly desired).
- Fix/clarify hybrid variant differentiation (loose/unconstrained currently not actually different).

---

### Date:
2026-04-04

### Task:
Repository cleanup and GitHub-oriented layout (no change to backtest mathematics)

### Files used:
- `src/` (all core modules moved here)
- `data/`, `results/`, `docs/`, `scripts/data_prep/`
- `main.py`, `export_metrics.py`, `merge_dataset.py`, `run_datacheck.py`
- `.gitignore`, `README.md`, `requirements.txt`

### What was done:
- organized code under `src/` with `src/paths.py` for repo-root-relative data/results paths
- moved `final_dataset.csv` → `data/final_dataset.csv`; pipeline outputs → `results/`
- removed `__pycache__`, `*.old`, and legacy duplicate folder with absolute-path data scripts
- added portable `scripts/data_prep/` (relative `data/raw/incoming/` inputs)
- added root launchers so `runpy`/`import` resolve `src` correctly; added `.gitignore`, `requirements.txt`, and root `README.md`

### Key result:
The project runs from the repo root with `python main.py` and `python export_metrics.py` after the layout change; **no absolute Windows paths remain in project Python sources**. Backtest logic inside strategy functions was not modified—only file locations and import/path plumbing.

### Action needed:
- Add Jupyter notebooks under `notebooks/` if desired; place thesis PDF under `docs/paper/` when publishing.