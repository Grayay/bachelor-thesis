"""Run exploratory data diagnostics (plots / statsmodels); see src/datacheck.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "src" / "datacheck.py"),
        run_name="__main__",
    )
