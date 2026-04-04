"""Paths for data-prep scripts (run from anywhere; resolves repo root)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Place vendor / broker export files here (see data/raw/README.md).
INCOMING = REPO_ROOT / "data" / "raw" / "incoming"
# Weekly return CSVs written here and consumed by src/merge.py.
OUT_WEEKLY = REPO_ROOT / "data" / "raw"
