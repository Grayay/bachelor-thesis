"""Export Chapter 3.1 descriptive dataset tables and figures.

Source of truth: data/final_dataset.csv.
Outputs:
- results/tables/table_descriptive_statistics_assets.csv
- results/tables/table_correlation_matrix_assets.csv
- results/figures/asset_returns_timeseries.png
- results/figures/asset_correlation_heatmap.png
- results/figures/asset_return_boxplot.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings(
    "ignore",
    message="The 'labels' parameter of boxplot\\(\\) has been renamed",
    category=matplotlib.MatplotlibDeprecationWarning,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "final_dataset.csv"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

DESCRIPTIVE_STATS_CSV = TABLES_DIR / "table_descriptive_statistics_assets.csv"
CORRELATION_CSV = TABLES_DIR / "table_correlation_matrix_assets.csv"
RETURNS_TIMESERIES_PNG = FIGURES_DIR / "asset_returns_timeseries.png"
CORRELATION_HEATMAP_PNG = FIGURES_DIR / "asset_correlation_heatmap.png"
BOXPLOT_PNG = FIGURES_DIR / "asset_return_boxplot.png"

ASSET_LABELS = {
    "btc_return": "BTC",
    "eth_return": "ETH",
    "gold_return": "Gold",
    "ruonia_return": "RUONIA",
    "moex_return": "MOEX",
    "ofz_return": "OFZ",
}

EXPECTED_COLUMNS = list(ASSET_LABELS.keys())
WEEKS_PER_YEAR = 52


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    df = df.loc[:, EXPECTED_COLUMNS]
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def build_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "asset": [ASSET_LABELS[c] for c in df.columns],
            "mean_weekly": df.mean().to_numpy(),
            "std_weekly": df.std(ddof=1).to_numpy(),
            "annualized_volatility": (df.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)).to_numpy(),
            "min_weekly": df.min().to_numpy(),
            "max_weekly": df.max().to_numpy(),
            "skewness": df.apply(lambda s: skew(s.dropna().to_numpy())).to_numpy(),
            "excess_kurtosis": df.apply(lambda s: kurtosis(s.dropna().to_numpy())).to_numpy(),
            "extreme_observations_abs_gt_30pct": (df.abs() > 0.30).sum().to_numpy(),
        }
    )
    return stats


def build_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.rename(columns=ASSET_LABELS)
    return labeled.corr()


def save_returns_timeseries(df: pd.DataFrame) -> None:
    labeled = df.rename(columns=ASSET_LABELS)
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()

    for ax, asset in zip(axes, labeled.columns):
        ax.plot(labeled.index, labeled[asset], linewidth=0.8, color="#1f77b4")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.55)
        ax.set_title(asset, fontsize=11)
        ax.set_ylabel("Log return")
        ax.grid(alpha=0.25)

    fig.suptitle("Weekly Asset Log Returns", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(RETURNS_TIMESERIES_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    labels = list(corr.index)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            color = "white" if abs(val) >= 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Correlation Matrix of Weekly Asset Log Returns", fontsize=13)
    fig.tight_layout()
    fig.savefig(CORRELATION_HEATMAP_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(df: pd.DataFrame) -> None:
    labeled = df.rename(columns=ASSET_LABELS)
    data = [labeled[c].to_numpy() for c in labeled.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        data,
        labels=list(labeled.columns),
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "#4c78a8",
            "markeredgecolor": "#4c78a8",
            "alpha": 0.65,
        },
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        boxprops={"color": "#444444", "linewidth": 1.0},
    )
    for box in bp["boxes"]:
        box.set_facecolor("#d6e4f0")
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.55)
    ax.set_title("Distribution of Weekly Asset Log Returns", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Weekly log return")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(BOXPLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(df: pd.DataFrame, stats: pd.DataFrame, corr: pd.DataFrame) -> None:
    generated = [
        DESCRIPTIVE_STATS_CSV,
        CORRELATION_CSV,
        RETURNS_TIMESERIES_PNG,
        CORRELATION_HEATMAP_PNG,
        BOXPLOT_PNG,
    ]

    print("Generated files:")
    for path in generated:
        print(f"- {path.relative_to(ROOT).as_posix()}")

    print("\nDescriptive statistics table:")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nCorrelation matrix:")
    print(corr.to_string(float_format=lambda x: f"{x:.4f}"))

    missing_values = int(df.isna().sum().sum())
    duplicate_dates = int(df.index.duplicated().sum())
    non_finite_values = int((~np.isfinite(df.to_numpy())).sum())

    print("\nSanity checks:")
    print(f"- shape: {df.shape}")
    print(f"- date range: {df.index.min().date()} .. {df.index.max().date()}")
    print(f"- missing values: {missing_values}")
    print(f"- duplicate dates: {duplicate_dates}")
    print(f"- non-finite values: {non_finite_values}")

    eth_vol = float(stats.loc[stats["asset"] == "ETH", "annualized_volatility"].iloc[0])
    btc_vol = float(stats.loc[stats["asset"] == "BTC", "annualized_volatility"].iloc[0])
    ruonia_vol = float(stats.loc[stats["asset"] == "RUONIA", "annualized_volatility"].iloc[0])
    btc_eth_corr = float(corr.loc["BTC", "ETH"])
    moex_ofz_corr = float(corr.loc["MOEX", "OFZ"])

    print("\nPreviously reported diagnostics match:")
    print(f"- ETH annualized volatility about 0.955: {np.isclose(eth_vol, 0.955, atol=0.001)} ({eth_vol:.6f})")
    print(f"- BTC annualized volatility about 0.725: {np.isclose(btc_vol, 0.725, atol=0.001)} ({btc_vol:.6f})")
    print(f"- BTC-ETH correlation about 0.682: {np.isclose(btc_eth_corr, 0.682, atol=0.001)} ({btc_eth_corr:.6f})")
    print(f"- MOEX-OFZ correlation about 0.574: {np.isclose(moex_ofz_corr, 0.574, atol=0.001)} ({moex_ofz_corr:.6f})")
    print(f"- RUONIA volatility about 0.0049: {np.isclose(ruonia_vol, 0.0049, atol=0.0001)} ({ruonia_vol:.6f})")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    stats = build_descriptive_stats(df)
    corr = build_correlation_matrix(df)

    stats.to_csv(DESCRIPTIVE_STATS_CSV, index=False, float_format="%.6f")
    corr.to_csv(CORRELATION_CSV, index=True, index_label="asset", float_format="%.4f")

    save_returns_timeseries(df)
    save_correlation_heatmap(corr)
    save_boxplot(df)

    print_report(df, stats, corr)


if __name__ == "__main__":
    main()
