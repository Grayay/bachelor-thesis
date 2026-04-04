"""
Entry point: run from repository root so `data/` and `results/` resolve correctly.

  python main.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from pipeline import run_pipeline

    run_pipeline()
