# Utility scripts

## `data_prep/`

Weekly log-return construction from vendor CSV/XLSX files. Paths are **relative to the repository root** via `paths_local.py`.

1. Place source files in `data/raw/incoming/` using the filenames expected by each script (see root `README.md`).
2. Run the desired script from its directory, for example:

```bash
cd scripts/data_prep
python btcrub.py
```

Intermediate weekly series are written to `data/raw/`.

## Root launchers

From the repository root, prefer:

- `python main.py` — full OOS backtest + `results/metrics.csv`
- `python export_metrics.py` — crisis / full-sample metric tables + L1 distances
- `python merge_dataset.py` — rebuild `data/final_dataset.csv` from `data/raw/*.csv`
- `python run_datacheck.py` — exploratory diagnostics (`src/datacheck.py`)
