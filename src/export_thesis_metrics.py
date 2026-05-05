from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd

from metrics import compute_metrics_table
from paths import (
    DIVERSIFICATION_RATIO_CSV,
    L1_DISTANCES_CSV,
    METRICS_CRISIS_CSV,
    METRICS_FULL_CSV,
    METRICS_NON_CRISIS_CSV,
    OOS_RETURNS_CSV,
)


def _read_oos_returns(path: str | None = None) -> pd.DataFrame:
    p = OOS_RETURNS_CSV if path is None else path
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


def _slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def _sanity_checks(metrics_df: pd.DataFrame) -> None:
    if metrics_df.isna().any().any():
        raise ValueError("Metrics contain NaNs.")

    vol = metrics_df["Volatility"]
    if not (vol > 0).all():
        raise ValueError("Volatility must be > 0 for all strategies.")

    # Soft checks (non-blocking) are intentionally omitted here to keep the
    # export script deterministic and non-interactive.


def _round_table(df: pd.DataFrame, decimals: int = 5) -> pd.DataFrame:
    return df.round(decimals)


def _l1_distance_table(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: pairwise mean absolute return differences.
    """
    cols = returns.columns
    diffs = returns[cols].to_numpy()[:, :, None] - returns[cols].to_numpy()[:, None, :]
    dist = np.nanmean(np.abs(diffs), axis=0)
    return pd.DataFrame(dist, index=cols, columns=cols)


def _read_diversification_ratio_panel() -> pd.DataFrame:
    dr = pd.read_csv(DIVERSIFICATION_RATIO_CSV, parse_dates=["rebalance_date"])
    if dr.empty:
        raise ValueError("Diversification ratio panel is empty.")
    if dr["diversification_ratio"].isna().any():
        raise ValueError("Diversification ratio panel contains NaN values.")
    return dr


def _add_dr_summary(metrics_tbl: pd.DataFrame, dr_panel: pd.DataFrame) -> pd.DataFrame:
    summary = dr_panel.groupby("strategy")["diversification_ratio"].agg(["mean", "median"])
    summary = summary.rename(
        columns={
            "mean": "DiversificationRatioMean",
            "median": "DiversificationRatioMedian",
        }
    )
    out = metrics_tbl.join(summary, how="left")
    if out[["DiversificationRatioMean", "DiversificationRatioMedian"]].isna().any().any():
        raise ValueError("Failed to join DR summaries for all strategies.")
    return out


if __name__ == "__main__":
    oos = _read_oos_returns()
    dr_panel = _read_diversification_ratio_panel()

    # Crisis definitions (weekly data; inclusive bounds).
    crisis_2020 = _slice_period(oos, "2020-01-01", "2020-12-31")
    crisis_2022 = _slice_period(oos, "2022-01-01", "2022-12-31")
    crisis = pd.concat([crisis_2020, crisis_2022]).sort_index()
    non_crisis = oos.drop(index=crisis.index, errors="ignore")

    full_tbl = compute_metrics_table(oos, rf=0.0, alpha=0.05)
    crisis_tbl = compute_metrics_table(crisis, rf=0.0, alpha=0.05)
    non_crisis_tbl = compute_metrics_table(non_crisis, rf=0.0, alpha=0.05)
    full_tbl = _add_dr_summary(full_tbl, dr_panel)
    crisis_dr = dr_panel.loc[
        dr_panel["rebalance_date"].between(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
        | dr_panel["rebalance_date"].between(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
    ]
    non_crisis_dr = dr_panel.loc[~dr_panel.index.isin(crisis_dr.index)]
    crisis_tbl = _add_dr_summary(crisis_tbl, crisis_dr)
    non_crisis_tbl = _add_dr_summary(non_crisis_tbl, non_crisis_dr)

    for tbl in (full_tbl, crisis_tbl, non_crisis_tbl):
        _sanity_checks(tbl)

    _round_table(full_tbl).to_csv(METRICS_FULL_CSV, index=True)
    _round_table(crisis_tbl).to_csv(METRICS_CRISIS_CSV, index=True)
    _round_table(non_crisis_tbl).to_csv(METRICS_NON_CRISIS_CSV, index=True)

    # Optional: L1 distance table
    _round_table(_l1_distance_table(oos), decimals=8).to_csv(L1_DISTANCES_CSV, index=True)
