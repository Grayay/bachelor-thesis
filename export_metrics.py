"""Regenerate crisis/full metrics tables and L1 distances (writes under results/)."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "src" / "export_thesis_metrics.py"),
        run_name="__main__",
    )
