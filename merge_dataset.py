"""Build data/final_dataset.csv from weekly asset CSVs in data/raw/."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "src" / "merge.py"),
        run_name="__main__",
    )
