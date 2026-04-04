# Russian Multi-Asset Portfolios with Crypto: OOS Backtests

Bachelor thesis (HSE) empirical code: rolling **out-of-sample** weekly portfolio strategies on a **RUB-denominated** panel (MOEX equity, OFZ bonds, gold, RUONIA, Bitcoin, Ethereum). Returns in the merged dataset are **weekly log returns**; portfolio OOS returns are stored as **portfolio log returns** after correct aggregation from asset simple returns.

## Research objective

Compare long-only allocation rules—**equal weight**, **long-only minimum variance** (sample and Ledoit–Wolf shrinkage covariance), **risk parity** (equal risk contribution), **fixed All-Weather weights**, and **All-Weather + risk parity hybrids**—under a **156-week** estimation window, **13-week** rebalancing, and optional **crypto caps**. Report performance and risk metrics (including **CVaR**, drawdown, Sharpe/Sortino) on the joint OOS period.

## Data

- **Merged panel:** `data/final_dataset.csv` (weekly log returns; columns described in code / thesis).
- **Raw vendor files:** not committed. Use `data/raw/incoming/` and the scripts under `scripts/data_prep/` (see `data/raw/README.md`).

## Methods (high level)

| Component | Description |
|-----------|-------------|
| Equal weight | Fixed \(1/N\) over the asset universe (base vs crypto-extended). |
| Minimum variance | Long-only MV from estimated covariance (sample or Ledoit–Wolf). |
| Risk parity | Long-only ERC via constrained optimisation (SLSQP). |
| All-Weather | Fixed strategic weights (base vs crypto-adjusted). |
| Hybrid | Convex mix of All-Weather anchors and ERC weights, with optional RUONIA cap variant. |
| Metrics | Volatility, Sharpe, Sortino, max drawdown, CVaR (5%), skewness, kurtosis; subsamples via `export_metrics.py`. |

## Repository structure

```
├── data/                  # Datasets (merged panel + raw README)
├── docs/                  # Project notes, issue log, thesis PDF placeholder (docs/paper/)
├── notebooks/             # Jupyter experiments (optional)
├── results/               # Generated OOS returns and metric tables (CSV)
├── scripts/               # Data prep and helpers (see scripts/README.md)
├── src/                   # Core backtest and metrics code
├── tests/                 # Small validation / sensitivity scripts
├── main.py                # Run full OOS backtest → results/
├── export_metrics.py      # Export full/crisis/non-crisis metrics + L1 table
├── merge_dataset.py       # Rebuild data/final_dataset.csv from data/raw/
├── run_datacheck.py       # Exploratory diagnostics
├── requirements.txt
└── README.md
```

Project memory / working notes: `docs/PROJECT_STATE.md`, `docs/CURRENT_ISSUES.md`, `docs/TODO_NEXT.md`, `docs/RESULTS_LOG.md`.

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt

# Full OOS backtest (writes results/oos_returns.csv, results/metrics.csv)
python main.py

# Additional metric slices + L1 distances
python export_metrics.py
```

**Rebuild `data/final_dataset.csv`:** place the six weekly CSVs listed in `data/raw/README.md` under `data/raw/`, then:

```bash
python merge_dataset.py
```

## Key results (brief)

After `main.py`, inspect `results/metrics.csv` and `results/oos_returns.csv`. Long-only **minimum variance with sample covariance** can allocate heavily to **low-volatility defensive legs** (e.g. RUONIA), which is a **model implication**, not necessarily a coding error. Hybrid variant rows may coincide if several functions share the same effective parameters—see `docs/RESULTS_LOG.md`.

## License / use

Academic portfolio demonstration. Data sources are user-provided; do not redistribute proprietary vendor files.
