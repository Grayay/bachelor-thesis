import numpy as np
import pandas as pd

# Canonical All-Weather backtest (used by root `main.py`).

from paths import FINAL_DATASET_CSV

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = str(FINAL_DATASET_CSV)
ROLLING_WINDOW_WEEKS = 156
REBALANCE_EVERY_WEEKS = 13

BASE_WEIGHTS = {
    "moex_return": 0.30,
    "ofz_return": 0.35,
    "gold_return": 0.20,
    "ruonia_return": 0.15,
}

CRYPTO_WEIGHTS = {
    "moex_return": 0.25,
    "ofz_return": 0.35,
    "gold_return": 0.20,
    "ruonia_return": 0.10,
    "btc_return": 0.05,
    "eth_return": 0.05,
}


def load_returns_dataset(path: str) -> pd.DataFrame:
    """
    Load weekly returns dataset from CSV, parse date, set as index.
    The `final_dataset.csv` used in this project contains weekly *log returns*.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def rolling_all_weather_backtest(
    returns: pd.DataFrame,
    weights_dict: dict[str, float],
    window: int = 156,
    step: int = 13,
    output: str = "log",
) -> pd.Series:
    """
    Rolling out-of-sample backtest for fixed-weight All-Weather portfolio.

    Input:
      - `returns`: weekly *log returns* per asset (as in `final_dataset.csv`)
    Output:
      - by default `output="log"`: portfolio log returns for each week in OOS blocks
      - if `output="simple"`: portfolio simple returns
    """
    if output not in {"log", "simple"}:
        raise ValueError("output must be either 'log' or 'simple'.")

    assets = list(weights_dict.keys())
    data_log = returns.loc[:, assets].dropna().copy()

    if len(data_log) <= window:
        raise ValueError(
            f"Not enough observations for backtest: have {len(data_log)}, need > {window}."
        )

    weights = np.array([weights_dict[a] for a in assets], dtype=float)
    if np.any(weights < 0):
        raise ValueError("All-Weather weights must be non-negative.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("All-Weather weights must sum to 1.")

    data_simple = np.expm1(data_log)

    oos_blocks: list[pd.Series] = []
    start = window
    n_obs = len(data_simple)

    while start < n_obs:
        end = min(start + step, n_obs)
        test_slice = data_simple.iloc[start:end]
        if test_slice.empty:
            break

        portfolio_simple = test_slice.to_numpy() @ weights
        portfolio = np.log1p(portfolio_simple) if output == "log" else portfolio_simple
        block_series = pd.Series(
            portfolio,
            index=test_slice.index,
            name="aw_returns",
        )
        oos_blocks.append(block_series)
        start += step

    if not oos_blocks:
        raise ValueError("Backtest produced no out-of-sample data.")

    oos = pd.concat(oos_blocks).sort_index()
    if oos.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in OOS output.")
    return oos

