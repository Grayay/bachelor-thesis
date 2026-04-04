"""
Repository-root-relative paths. All scripts should use these instead of absolute paths.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = REPO_ROOT / "results"

FINAL_DATASET_CSV = DATA_DIR / "final_dataset.csv"
OOS_RETURNS_CSV = RESULTS_DIR / "oos_returns.csv"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
METRICS_FULL_CSV = RESULTS_DIR / "metrics_full.csv"
METRICS_CRISIS_CSV = RESULTS_DIR / "metrics_crisis.csv"
METRICS_NON_CRISIS_CSV = RESULTS_DIR / "metrics_non_crisis.csv"
L1_DISTANCES_CSV = RESULTS_DIR / "l1_distances.csv"
