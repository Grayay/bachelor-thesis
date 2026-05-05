"""
Generate thesis-ready diagnostics, tables, plots, and reproducibility outputs.

This script is intentionally read-only with respect to core methodology:
- no strategy definition changes;
- no optimizer logic changes;
- no data changes.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allweather import BASE_WEIGHTS, CRYPTO_WEIGHTS  # noqa: E402
from paths import (  # noqa: E402
    DIVERSIFICATION_RATIO_CSV,
    HYBRID_DIAGNOSTICS_CSV,
    METRICS_CRISIS_CSV,
    METRICS_CSV,
    METRICS_FULL_CSV,
    METRICS_NON_CRISIS_CSV,
    OOS_RETURNS_CSV,
    RESULTS_DIR,
)
from riskparity import (  # noqa: E402
    BASE_ASSETS,
    CRYPTO_ASSETS,
    CSV_PATH,
    REBALANCE_EVERY_WEEKS,
    ROLLING_WINDOW_WEEKS,
    compute_min_variance_weights,
    compute_risk_parity_weights,
    load_returns_dataset,
)
from rpallw import BASE_CAPS, CRYPTO_CAPS, compute_hybrid_weights  # noqa: E402


FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
WEIGHTS_HISTORY_CSV = RESULTS_DIR / "weights_history.csv"
RISK_CONTRIB_CSV = RESULTS_DIR / "risk_contributions.csv"
WEIGHTS_SUMMARY_CSV = RESULTS_DIR / "weights_summary.csv"
CRYPTO_EXPOSURE_SUMMARY_CSV = RESULTS_DIR / "crypto_exposure_summary.csv"
RUONIA_EXPOSURE_SUMMARY_CSV = RESULTS_DIR / "ruonia_exposure_summary.csv"
RISK_CONTRIB_SUMMARY_CSV = RESULTS_DIR / "risk_contribution_summary.csv"
CRYPTO_RISK_CONTRIB_SUMMARY_CSV = RESULTS_DIR / "crypto_risk_contribution_summary.csv"
RUONIA_RISK_CONTRIB_SUMMARY_CSV = RESULTS_DIR / "ruonia_risk_contribution_summary.csv"
ENB_CSV = RESULTS_DIR / "enb.csv"
ENB_SUMMARY_CSV = RESULTS_DIR / "enb_summary.csv"
VALIDATION_REPORT_MD = RESULTS_DIR / "EXPORT_VALIDATION_REPORT.md"


@dataclass(frozen=True)
class StrategySpec:
    strategy: str
    universe: str
    assets: list[str]
    cov_type: str
    weight_fn: Callable[[np.ndarray], np.ndarray]


def _estimate_covariance(train_slice: pd.DataFrame, cov_type: str) -> np.ndarray:
    if cov_type == "sample":
        return train_slice.cov().to_numpy()
    if cov_type == "shrinkage":
        lw = LedoitWolf()
        lw.fit(train_slice.to_numpy())
        return lw.covariance_
    raise ValueError(f"Unsupported cov_type: {cov_type}")


def _strategy_specs() -> list[StrategySpec]:
    base_aw_assets = list(BASE_WEIGHTS.keys())
    crypto_aw_assets = list(CRYPTO_WEIGHTS.keys())
    base_h_assets = list(BASE_CAPS.keys())
    crypto_h_assets = list(CRYPTO_CAPS.keys())
    return [
        StrategySpec("ew_base", "base", BASE_ASSETS, "sample", lambda cov: np.full(len(BASE_ASSETS), 1.0 / len(BASE_ASSETS))),
        StrategySpec("ew_crypto", "crypto", CRYPTO_ASSETS, "sample", lambda cov: np.full(len(CRYPTO_ASSETS), 1.0 / len(CRYPTO_ASSETS))),
        StrategySpec("mv_sample_base", "base", BASE_ASSETS, "sample", compute_min_variance_weights),
        StrategySpec("mv_sample_crypto", "crypto", CRYPTO_ASSETS, "sample", compute_min_variance_weights),
        StrategySpec("mv_shrink_base", "base", BASE_ASSETS, "shrinkage", compute_min_variance_weights),
        StrategySpec("mv_shrink_crypto", "crypto", CRYPTO_ASSETS, "shrinkage", compute_min_variance_weights),
        StrategySpec("rp_sample_base", "base", BASE_ASSETS, "sample", compute_risk_parity_weights),
        StrategySpec("rp_sample_crypto", "crypto", CRYPTO_ASSETS, "sample", compute_risk_parity_weights),
        StrategySpec("rp_shrink_base", "base", BASE_ASSETS, "shrinkage", compute_risk_parity_weights),
        StrategySpec("rp_shrink_crypto", "crypto", CRYPTO_ASSETS, "shrinkage", compute_risk_parity_weights),
        StrategySpec("aw_base", "base", base_aw_assets, "sample", lambda cov: np.array([BASE_WEIGHTS[a] for a in base_aw_assets], dtype=float)),
        StrategySpec("aw_crypto", "crypto", crypto_aw_assets, "sample", lambda cov: np.array([CRYPTO_WEIGHTS[a] for a in crypto_aw_assets], dtype=float)),
        StrategySpec("hybrid_base", "base", base_h_assets, "sample", lambda cov: compute_hybrid_weights(cov, BASE_CAPS, regime="constrained")[1]),
        StrategySpec("hybrid_crypto", "crypto", crypto_h_assets, "sample", lambda cov: compute_hybrid_weights(cov, CRYPTO_CAPS, regime="constrained")[1]),
        StrategySpec("hybrid_loose_base", "base", base_h_assets, "sample", lambda cov: compute_hybrid_weights(cov, BASE_CAPS, regime="loose")[1]),
        StrategySpec("hybrid_loose_crypto", "crypto", crypto_h_assets, "sample", lambda cov: compute_hybrid_weights(cov, CRYPTO_CAPS, regime="loose")[1]),
        StrategySpec("hybrid_unconstrained_base", "base", base_h_assets, "sample", lambda cov: compute_hybrid_weights(cov, BASE_CAPS, regime="unconstrained")[1]),
        StrategySpec("hybrid_unconstrained_crypto", "crypto", crypto_h_assets, "sample", lambda cov: compute_hybrid_weights(cov, CRYPTO_CAPS, regime="unconstrained")[1]),
        StrategySpec("hybrid_ruonia_capped_base", "base", base_h_assets, "sample", lambda cov: compute_hybrid_weights(cov, BASE_CAPS, regime="constrained", ruonia_cap=0.25)[1]),
        StrategySpec("hybrid_ruonia_capped_crypto", "crypto", crypto_h_assets, "sample", lambda cov: compute_hybrid_weights(cov, CRYPTO_CAPS, regime="constrained", ruonia_cap=0.25)[1]),
    ]


def _compute_weights_and_rc(returns_df: pd.DataFrame, specs: list[StrategySpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    weight_rows: list[dict[str, object]] = []
    rc_rows: list[dict[str, object]] = []
    tol = 1e-8

    for spec in specs:
        data = returns_df.loc[:, spec.assets].dropna().copy()
        start = ROLLING_WINDOW_WEEKS
        n_obs = len(data)

        while start < n_obs:
            train = data.iloc[start - ROLLING_WINDOW_WEEKS : start]
            if len(train) != ROLLING_WINDOW_WEEKS:
                break
            cov = _estimate_covariance(train, spec.cov_type)
            w = np.asarray(spec.weight_fn(cov), dtype=float)
            if len(w) != len(spec.assets):
                raise ValueError(f"Weight length mismatch for {spec.strategy}.")
            if not np.isfinite(w).all():
                raise ValueError(f"Non-finite weights for {spec.strategy}.")
            if np.any(w < -tol):
                raise ValueError(f"Negative weights for {spec.strategy}.")
            ws = w.sum()
            if not np.isclose(ws, 1.0, atol=1e-6):
                raise ValueError(f"Weights do not sum to 1 for {spec.strategy} at {data.index[start]}.")

            rebalance_date = data.index[start]
            sigma_w = cov @ w
            port_var = float(w.T @ cov @ w)
            if port_var <= 0:
                raise ValueError(f"Non-positive portfolio variance for {spec.strategy} at {rebalance_date}.")
            port_vol = float(np.sqrt(port_var))
            mrc = sigma_w / port_vol
            rc = w * mrc
            rc_share = rc / port_vol

            for i, asset in enumerate(spec.assets):
                weight_rows.append(
                    {
                        "rebalance_date": rebalance_date,
                        "strategy": spec.strategy,
                        "universe": spec.universe,
                        "asset": asset,
                        "weight": float(w[i]),
                    }
                )
                rc_rows.append(
                    {
                        "rebalance_date": rebalance_date,
                        "strategy": spec.strategy,
                        "universe": spec.universe,
                        "asset": asset,
                        "weight": float(w[i]),
                        "mrc": float(mrc[i]),
                        "rc": float(rc[i]),
                        "rc_share": float(rc_share[i]),
                    }
                )
            start += REBALANCE_EVERY_WEEKS

    weights_df = pd.DataFrame(weight_rows).sort_values(["rebalance_date", "strategy", "asset"]).reset_index(drop=True)
    rc_df = pd.DataFrame(rc_rows).sort_values(["rebalance_date", "strategy", "asset"]).reset_index(drop=True)
    return weights_df, rc_df


def _sanity_checks(weights_df: pd.DataFrame, rc_df: pd.DataFrame) -> dict[str, list[str]]:
    checks: dict[str, list[str]] = {"passed": [], "failed": []}
    tol = 1e-6

    sum_w = weights_df.groupby(["rebalance_date", "strategy"])["weight"].sum()
    if np.allclose(sum_w.values, 1.0, atol=tol):
        checks["passed"].append("weights sum to 1 by rebalance_date×strategy")
    else:
        checks["failed"].append("weights sum to 1 by rebalance_date×strategy")

    if (weights_df["weight"] >= -1e-10).all():
        checks["passed"].append("no negative weights")
    else:
        checks["failed"].append("no negative weights")

    if not weights_df["weight"].isna().any():
        checks["passed"].append("no missing weights")
    else:
        checks["failed"].append("no missing weights")

    base_has_crypto = weights_df.query("universe == 'base' and asset in ['btc_return','eth_return']")
    if base_has_crypto.empty:
        checks["passed"].append("base strategies do not include BTC/ETH")
    else:
        checks["failed"].append("base strategies do not include BTC/ETH")

    crypto = weights_df.query("universe == 'crypto'")
    strategy_assets = crypto.groupby("strategy")["asset"].agg(set)
    ok_crypto = all({"btc_return", "eth_return"}.issubset(assets) for assets in strategy_assets)
    if ok_crypto:
        checks["passed"].append("crypto strategies include BTC/ETH")
    else:
        checks["failed"].append("crypto strategies include BTC/ETH")

    finite_rc = np.isfinite(rc_df[["mrc", "rc", "rc_share"]].to_numpy()).all()
    if finite_rc:
        checks["passed"].append("risk contribution values are finite")
    else:
        checks["failed"].append("risk contribution values are finite")

    rc_share_sum = rc_df.groupby(["rebalance_date", "strategy"])["rc_share"].sum()
    if np.allclose(rc_share_sum.values, 1.0, atol=1e-4):
        checks["passed"].append("sum(rc_share) approximately 1 by rebalance_date×strategy")
    else:
        checks["failed"].append("sum(rc_share) approximately 1 by rebalance_date×strategy")

    return checks


def _save_summaries(weights_df: pd.DataFrame, rc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights_summary = (
        weights_df.groupby(["strategy", "asset"])["weight"]
        .agg(mean_weight="mean", median_weight="median", min_weight="min", max_weight="max")
        .reset_index()
    )
    weights_summary.to_csv(WEIGHTS_SUMMARY_CSV, index=False)

    crypto_strategies = sorted(weights_df.loc[weights_df["universe"] == "crypto", "strategy"].unique())
    crypto_rows: list[dict[str, object]] = []
    for strategy in crypto_strategies:
        subset = weights_df.loc[weights_df["strategy"] == strategy]
        btc = subset.loc[subset["asset"] == "btc_return", "weight"]
        eth = subset.loc[subset["asset"] == "eth_return", "weight"]
        crypto_rows.append(
            {
                "strategy": strategy,
                "mean_btc_weight": float(btc.mean()),
                "max_btc_weight": float(btc.max()),
                "mean_eth_weight": float(eth.mean()),
                "max_eth_weight": float(eth.max()),
                "share_dates_btc_gt_0": float((btc > 0).mean()),
                "share_dates_eth_gt_0": float((eth > 0).mean()),
            }
        )
    crypto_summary = pd.DataFrame(crypto_rows)
    crypto_summary.to_csv(CRYPTO_EXPOSURE_SUMMARY_CSV, index=False)

    ruonia = weights_df.loc[weights_df["asset"] == "ruonia_return"]
    ruonia_summary = (
        ruonia.groupby("strategy")["weight"]
        .agg(mean_ruonia_weight="mean", median_ruonia_weight="median", max_ruonia_weight="max")
        .reset_index()
    )
    # Use tolerance to avoid floating-point artifacts around the 25% cap.
    share25 = ruonia.groupby("strategy")["weight"].apply(lambda s: float((s > 0.250001).mean())).rename("share_dates_ruonia_gt_25")
    share50 = ruonia.groupby("strategy")["weight"].apply(lambda s: float((s > 0.50).mean())).rename("share_dates_ruonia_gt_50")
    ruonia_summary = ruonia_summary.merge(share25.reset_index(), on="strategy").merge(share50.reset_index(), on="strategy")
    ruonia_summary["ruonia_cap_25_bind_share"] = np.where(
        ruonia_summary["strategy"].str.contains("hybrid_ruonia_capped"),
        ruonia.groupby("strategy")["weight"].apply(lambda s: float(np.isclose(s, 0.25, atol=1e-6).mean())).values,
        np.nan,
    )
    ruonia_summary.to_csv(RUONIA_EXPOSURE_SUMMARY_CSV, index=False)

    rc_summary = (
        rc_df.groupby(["strategy", "asset"])["rc_share"]
        .agg(mean_rc_share="mean", median_rc_share="median", min_rc_share="min", max_rc_share="max")
        .reset_index()
    )
    rc_summary.to_csv(RISK_CONTRIB_SUMMARY_CSV, index=False)

    crypto_rc_rows: list[dict[str, object]] = []
    for strategy in crypto_strategies:
        subset = rc_df.loc[rc_df["strategy"] == strategy]
        btc = subset.loc[subset["asset"] == "btc_return", "rc_share"]
        eth = subset.loc[subset["asset"] == "eth_return", "rc_share"]
        combined = btc.to_numpy() + eth.to_numpy()
        crypto_rc_rows.append(
            {
                "strategy": strategy,
                "mean_btc_rc_share": float(btc.mean()),
                "max_btc_rc_share": float(btc.max()),
                "mean_eth_rc_share": float(eth.mean()),
                "max_eth_rc_share": float(eth.max()),
                "combined_mean_crypto_rc_share": float(combined.mean()),
                "combined_max_crypto_rc_share": float(combined.max()),
            }
        )
    crypto_rc_summary = pd.DataFrame(crypto_rc_rows)
    crypto_rc_summary.to_csv(CRYPTO_RISK_CONTRIB_SUMMARY_CSV, index=False)

    ruonia_rc_summary = (
        rc_df.loc[rc_df["asset"] == "ruonia_return"]
        .groupby("strategy")["rc_share"]
        .agg(mean_ruonia_rc_share="mean", max_ruonia_rc_share="max")
        .reset_index()
    )
    ruonia_rc_summary.to_csv(RUONIA_RISK_CONTRIB_SUMMARY_CSV, index=False)
    return crypto_summary, ruonia_summary


def _save_enb(rc_df: pd.DataFrame) -> pd.DataFrame:
    enb = (
        rc_df.groupby(["rebalance_date", "strategy"])["rc_share"]
        .apply(lambda s: float(1.0 / np.sum(np.square(s.to_numpy()))))
        .rename("enb")
        .reset_index()
    )
    enb.to_csv(ENB_CSV, index=False)
    enb_summary = (
        enb.groupby("strategy")["enb"]
        .agg(enb_mean="mean", enb_median="median", enb_min="min", enb_max="max")
        .reset_index()
    )
    enb_summary.to_csv(ENB_SUMMARY_CSV, index=False)
    return enb_summary


def _best_worst_table(metrics_full: pd.DataFrame, enb_summary: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for metric in ["CAGR", "Volatility", "Sharpe", "Sortino", "CVaR_5%", "MaxDrawdown", "DiversificationRatioMean"]:
        s = metrics_full[metric]
        if metric == "Volatility":
            best_idx, worst_idx = s.idxmin(), s.idxmax()
        elif metric in {"CVaR_5%", "MaxDrawdown"}:
            best_idx, worst_idx = s.idxmax(), s.idxmin()
        else:
            best_idx, worst_idx = s.idxmax(), s.idxmin()
        rows.append(
            {
                "metric": metric,
                "best_strategy": best_idx,
                "best_value": float(s.loc[best_idx]),
                "worst_strategy": worst_idx,
                "worst_value": float(s.loc[worst_idx]),
            }
        )
    if enb_summary is not None and not enb_summary.empty:
        s = enb_summary.set_index("strategy")["enb_mean"]
        rows.append(
            {
                "metric": "ENB_RC_Proxy",
                "best_strategy": s.idxmax(),
                "best_value": float(s.max()),
                "worst_strategy": s.idxmin(),
                "worst_value": float(s.min()),
            }
        )
    return pd.DataFrame(rows)


def _base_vs_crypto_table(metrics_full: pd.DataFrame) -> pd.DataFrame:
    families = [
        ("ew", "ew_base", "ew_crypto"),
        ("mv_sample", "mv_sample_base", "mv_sample_crypto"),
        ("mv_shrink", "mv_shrink_base", "mv_shrink_crypto"),
        ("rp_sample", "rp_sample_base", "rp_sample_crypto"),
        ("rp_shrink", "rp_shrink_base", "rp_shrink_crypto"),
        ("aw", "aw_base", "aw_crypto"),
        ("hybrid", "hybrid_base", "hybrid_crypto"),
        ("hybrid_loose", "hybrid_loose_base", "hybrid_loose_crypto"),
        ("hybrid_unconstrained", "hybrid_unconstrained_base", "hybrid_unconstrained_crypto"),
        ("hybrid_ruonia_capped", "hybrid_ruonia_capped_base", "hybrid_ruonia_capped_crypto"),
    ]
    rows = []
    cols = ["CAGR", "Volatility", "Sharpe", "Sortino", "CVaR_5%", "MaxDrawdown", "DiversificationRatioMean", "DiversificationRatioMedian"]
    for fam, b, c in families:
        d = metrics_full.loc[c, cols] - metrics_full.loc[b, cols]
        row = {"family": fam, "base": b, "crypto": c}
        row.update({f"delta_{k}": float(v) for k, v in d.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_shortlist_table(metrics_full: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "rp_shrink_base",
        "rp_shrink_crypto",
        "aw_crypto",
        "hybrid_ruonia_capped_crypto",
        "mv_shrink_crypto",
    ]
    text = {
        "rp_shrink_base": ("Strong shrinkage RP baseline", "Stable risk budget", "Lower CAGR than crypto peers"),
        "rp_shrink_crypto": ("High DR and strong CAGR", "Strong diversification profile", "Tail risk and drawdown higher"),
        "aw_crypto": ("Core thesis-consistent extended AW", "Return and Sharpe uplift", "Crisis tail risk worsens"),
        "hybrid_ruonia_capped_crypto": ("Cap-aware hybrid candidate", "Balances anchor and RP blend", "Needs cap-binding diagnostics context"),
        "mv_shrink_crypto": ("Constrained MV with shrinkage", "Controlled volatility", "Can still concentrate exposures"),
    }
    rows = []
    for s in candidates:
        rows.append(
            {
                "strategy": s,
                "why_candidate": text[s][0],
                "main_strength": text[s][1],
                "main_weakness": text[s][2],
                "risk_return_comment": f"CAGR={metrics_full.loc[s, 'CAGR']:.4f}, Sharpe={metrics_full.loc[s, 'Sharpe']:.4f}, MaxDD={metrics_full.loc[s, 'MaxDrawdown']:.4f}",
                "diversification_comment": f"DR mean={metrics_full.loc[s, 'DiversificationRatioMean']:.4f}",
            }
        )
    return pd.DataFrame(rows)


def _save_tables(
    metrics_full: pd.DataFrame,
    metrics_crisis: pd.DataFrame,
    metrics_non_crisis: pd.DataFrame,
    weights_df: pd.DataFrame,
    crypto_summary: pd.DataFrame,
    ruonia_summary: pd.DataFrame,
    rc_summary: pd.DataFrame,
    enb_summary: pd.DataFrame,
) -> None:
    def _for_thesis_display(df: pd.DataFrame, include_full_dr_name: bool) -> pd.DataFrame:
        out = df.copy()
        if include_full_dr_name and "DiversificationRatio" in out.columns:
            out = out.rename(columns={"DiversificationRatio": "FullSampleDiversificationRatio"})
        if "Sortino" in out.columns:
            out["Sortino"] = out["Sortino"].apply(
                lambda x: "N/A" if isinstance(x, (float, np.floating)) and not np.isfinite(x) else x
            )
        return out

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _for_thesis_display(metrics_full, include_full_dr_name=False).to_csv(TABLES_DIR / "table_metrics_full.csv")
    _for_thesis_display(metrics_crisis, include_full_dr_name=True).to_csv(TABLES_DIR / "table_metrics_crisis.csv")
    _for_thesis_display(metrics_non_crisis, include_full_dr_name=True).to_csv(TABLES_DIR / "table_metrics_non_crisis.csv")

    base_crypto = _base_vs_crypto_table(metrics_full)
    base_crypto.to_csv(TABLES_DIR / "table_base_vs_crypto_deltas.csv", index=False)

    best_worst = _best_worst_table(metrics_full, enb_summary)
    best_worst.to_csv(TABLES_DIR / "table_best_worst_by_metric.csv", index=False)

    hybrid_strats = [
        "hybrid_base",
        "hybrid_loose_base",
        "hybrid_unconstrained_base",
        "hybrid_ruonia_capped_base",
        "hybrid_crypto",
        "hybrid_loose_crypto",
        "hybrid_unconstrained_crypto",
        "hybrid_ruonia_capped_crypto",
    ]
    _for_thesis_display(metrics_full.loc[hybrid_strats], include_full_dr_name=False).to_csv(
        TABLES_DIR / "table_hybrid_comparison.csv"
    )

    table_crypto_exposure = (
        weights_df.loc[weights_df["universe"] == "crypto"]
        .groupby(["strategy", "asset"])["weight"]
        .mean()
        .rename("mean_weight")
        .reset_index()
        .merge(crypto_summary, on="strategy", how="left")
    )
    table_crypto_exposure.to_csv(TABLES_DIR / "table_crypto_exposure.csv", index=False)

    table_ruonia_exposure = ruonia_summary.copy()
    table_ruonia_exposure.to_csv(TABLES_DIR / "table_ruonia_exposure.csv", index=False)

    rc_summary.to_csv(TABLES_DIR / "table_risk_contributions.csv", index=False)

    shortlist = _candidate_shortlist_table(metrics_full)
    shortlist.to_csv(TABLES_DIR / "table_final_candidate_shortlist.csv", index=False)


def _plot_save(fig_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


def _plot_cumulative(oos: pd.DataFrame, cols: list[str], title: str, path: Path) -> None:
    plt.figure(figsize=(14, 8))
    for c in cols:
        plt.plot(oos.index, np.exp(oos[c].cumsum()), label=c, linewidth=1.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of 1 RUB")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    _plot_save(path)


def _plot_drawdowns(oos: pd.DataFrame, cols: list[str], title: str, path: Path) -> None:
    plt.figure(figsize=(14, 8))
    for c in cols:
        eq = np.exp(oos[c].cumsum())
        dd = eq / eq.cummax() - 1
        plt.plot(oos.index, dd, label=c, linewidth=1.4)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    _plot_save(path)


def _plot_metric_bar(metrics_full: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    s = metrics_full[metric].sort_values(ascending=False)
    plt.figure(figsize=(13, 6))
    plt.bar(s.index, s.values)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=70, ha="right")
    plt.grid(axis="y", alpha=0.25)
    _plot_save(path)


def _plot_dr(metrics_full: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(metrics_full.index))
    w = 0.38
    plt.figure(figsize=(13, 6))
    plt.bar(x - w / 2, metrics_full["DiversificationRatioMean"].values, w, label="Mean")
    plt.bar(x + w / 2, metrics_full["DiversificationRatioMedian"].values, w, label="Median")
    plt.title("Diversification ratio by strategy")
    plt.ylabel("Diversification ratio")
    plt.xticks(x, metrics_full.index, rotation=70, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _plot_save(path)


def _plot_enb(enb_summary: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(enb_summary))
    w = 0.38
    plt.figure(figsize=(13, 6))
    plt.bar(x - w / 2, enb_summary["enb_mean"], w, label="Mean")
    plt.bar(x + w / 2, enb_summary["enb_median"], w, label="Median")
    plt.title("ENB by strategy (risk-contribution-based proxy)")
    plt.ylabel("ENB")
    plt.xticks(x, enb_summary["strategy"], rotation=70, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _plot_save(path)


def _plot_weights(weights_df: pd.DataFrame, strategy: str, path: Path) -> None:
    subset = weights_df.loc[weights_df["strategy"] == strategy].copy()
    pivot = subset.pivot(index="rebalance_date", columns="asset", values="weight").sort_index()
    plt.figure(figsize=(14, 7))
    plt.stackplot(pivot.index, [pivot[c].values for c in pivot.columns], labels=list(pivot.columns), alpha=0.9)
    plt.title(f"Weights over time: {strategy}")
    plt.xlabel("Rebalance date")
    plt.ylabel("Weight")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(alpha=0.2)
    _plot_save(path)


def _plot_lines_by_asset(weights_df: pd.DataFrame, asset: str, title: str, path: Path) -> None:
    subset = weights_df.loc[weights_df["asset"] == asset]
    plt.figure(figsize=(14, 7))
    for strategy, grp in subset.groupby("strategy"):
        plt.plot(grp["rebalance_date"], grp["weight"], label=strategy, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Rebalance date")
    plt.ylabel("Weight")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.25)
    _plot_save(path)


def _plot_rc_selected(rc_df: pd.DataFrame, strategy: str, path: Path) -> None:
    subset = rc_df.loc[rc_df["strategy"] == strategy].copy()
    pivot = subset.pivot(index="rebalance_date", columns="asset", values="rc_share").sort_index()
    plt.figure(figsize=(14, 7))
    for c in pivot.columns:
        plt.plot(pivot.index, pivot[c], label=c, linewidth=1.4)
    plt.title(f"Risk contribution shares over time: {strategy}")
    plt.xlabel("Rebalance date")
    plt.ylabel("RC share")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    _plot_save(path)


def _plot_crisis_vs_non(metrics_full: pd.DataFrame, metrics_crisis: pd.DataFrame, metrics_non: pd.DataFrame, metric: str, path: Path) -> None:
    selected = [
        "ew_base",
        "ew_crypto",
        "mv_shrink_base",
        "mv_shrink_crypto",
        "rp_shrink_base",
        "rp_shrink_crypto",
        "aw_base",
        "aw_crypto",
        "hybrid_base",
        "hybrid_crypto",
        "hybrid_ruonia_capped_base",
        "hybrid_ruonia_capped_crypto",
    ]
    x = np.arange(len(selected))
    w = 0.25
    plt.figure(figsize=(15, 7))
    plt.bar(x - w, metrics_full.loc[selected, metric], w, label="Full")
    plt.bar(x, metrics_crisis.loc[selected, metric], w, label="Crisis")
    plt.bar(x + w, metrics_non.loc[selected, metric], w, label="Non-crisis")
    plt.title(f"{metric}: full vs crisis vs non-crisis")
    plt.ylabel(metric)
    plt.xticks(x, selected, rotation=70, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _plot_save(path)


def _save_figures(
    oos: pd.DataFrame,
    weights_df: pd.DataFrame,
    rc_df: pd.DataFrame,
    metrics_full: pd.DataFrame,
    metrics_crisis: pd.DataFrame,
    metrics_non: pd.DataFrame,
    enb_summary: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    selected = [
        "ew_base",
        "ew_crypto",
        "mv_shrink_base",
        "mv_shrink_crypto",
        "rp_shrink_base",
        "rp_shrink_crypto",
        "aw_base",
        "aw_crypto",
        "hybrid_base",
        "hybrid_crypto",
        "hybrid_ruonia_capped_base",
        "hybrid_ruonia_capped_crypto",
    ]
    _plot_cumulative(oos, selected, "Cumulative OOS returns (selected)", FIGURES_DIR / "cumulative_returns_selected.png")
    _plot_drawdowns(oos, selected, "Drawdown curves (selected)", FIGURES_DIR / "drawdowns_selected.png")
    _plot_metric_bar(metrics_full, "Sharpe", "Sharpe by strategy (full)", FIGURES_DIR / "full_metrics_bar_sharpe.png")
    _plot_metric_bar(metrics_full, "CVaR_5%", "CVaR 5% by strategy (full)", FIGURES_DIR / "full_metrics_bar_cvar.png")
    _plot_metric_bar(metrics_full, "MaxDrawdown", "Max drawdown by strategy (full)", FIGURES_DIR / "full_metrics_bar_maxdd.png")
    _plot_dr(metrics_full, FIGURES_DIR / "diversification_ratio_by_strategy.png")
    _plot_enb(enb_summary, FIGURES_DIR / "enb_by_strategy.png")

    hybrid_cols = [
        "hybrid_base",
        "hybrid_loose_base",
        "hybrid_unconstrained_base",
        "hybrid_ruonia_capped_base",
        "hybrid_crypto",
        "hybrid_loose_crypto",
        "hybrid_unconstrained_crypto",
        "hybrid_ruonia_capped_crypto",
    ]
    _plot_cumulative(oos, hybrid_cols, "Hybrid family cumulative returns", FIGURES_DIR / "hybrid_comparison_cumulative.png")
    _plot_metric_bar(
        metrics_full.loc[hybrid_cols],
        "Sharpe",
        "Hybrid variants: Sharpe (full)",
        FIGURES_DIR / "hybrid_comparison_metrics.png",
    )

    for s in ["hybrid_crypto", "hybrid_ruonia_capped_crypto", "rp_shrink_crypto", "aw_crypto", "mv_shrink_crypto"]:
        _plot_weights(weights_df, s, FIGURES_DIR / f"weights_{s}.png")
    _plot_weights(weights_df, "hybrid_crypto", FIGURES_DIR / "weights_dynamics_selected.png")

    crypto_w = weights_df.loc[weights_df["asset"].isin(["btc_return", "eth_return"])]
    plt.figure(figsize=(14, 7))
    for (strategy, asset), grp in crypto_w.groupby(["strategy", "asset"]):
        plt.plot(grp["rebalance_date"], grp["weight"], label=f"{strategy}:{asset}", linewidth=1.2)
    plt.title("Crypto weights over time")
    plt.xlabel("Rebalance date")
    plt.ylabel("Weight")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.25)
    _plot_save(FIGURES_DIR / "crypto_weights_over_time.png")

    _plot_lines_by_asset(weights_df, "ruonia_return", "RUONIA weights over time", FIGURES_DIR / "ruonia_weights_over_time.png")

    for s in ["hybrid_crypto", "hybrid_ruonia_capped_crypto", "rp_shrink_crypto"]:
        _plot_rc_selected(rc_df, s, FIGURES_DIR / f"risk_contributions_{s}.png")
    _plot_rc_selected(rc_df, "hybrid_crypto", FIGURES_DIR / "risk_contributions_selected.png")

    crypto_rc = rc_df.loc[rc_df["asset"].isin(["btc_return", "eth_return"])]
    plt.figure(figsize=(14, 7))
    for (strategy, asset), grp in crypto_rc.groupby(["strategy", "asset"]):
        plt.plot(grp["rebalance_date"], grp["rc_share"], label=f"{strategy}:{asset}", linewidth=1.2)
    plt.title("Crypto risk contribution shares over time")
    plt.xlabel("Rebalance date")
    plt.ylabel("RC share")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.25)
    _plot_save(FIGURES_DIR / "crypto_risk_contribution.png")

    _plot_crisis_vs_non(metrics_full, metrics_crisis, metrics_non, "Sharpe", FIGURES_DIR / "crisis_vs_noncrisis_sharpe.png")
    _plot_crisis_vs_non(metrics_full, metrics_crisis, metrics_non, "CVaR_5%", FIGURES_DIR / "crisis_vs_noncrisis_cvar.png")


def _topk(series: pd.Series, higher_is_better: bool = True, k: int = 5) -> pd.Series:
    return series.sort_values(ascending=not higher_is_better).head(k)


def _write_validation_report(
    returns_df: pd.DataFrame,
    oos: pd.DataFrame,
    weights_df: pd.DataFrame,
    rc_df: pd.DataFrame,
    dr_df: pd.DataFrame,
    enb_df: pd.DataFrame,
    metrics_full: pd.DataFrame,
    checks: dict[str, list[str]],
) -> None:
    generated = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in RESULTS_DIR.rglob("*") if p.is_file())
    strategies = list(oos.columns)

    top_sharpe = _topk(metrics_full["Sharpe"], higher_is_better=True)
    top_cvar = _topk(metrics_full["CVaR_5%"], higher_is_better=True)
    top_dr = _topk(metrics_full["DiversificationRatioMean"], higher_is_better=True)
    top_enb = _topk(enb_df.groupby("strategy")["enb"].mean(), higher_is_better=True)

    lines = [
        "# Export Validation Report",
        "",
        "## Reproducibility commands",
        "- `python main.py`",
        "- `python export_metrics.py`",
        "- `python export_thesis_artifacts.py`",
        "",
        "## Generated files",
    ]
    lines.extend([f"- `{g}`" for g in generated])
    lines.extend(
        [
            "",
            "## Final strategy list",
            ", ".join(strategies),
            "",
            "## Shapes and ranges",
            f"- Dataset shape: {returns_df.shape}",
            f"- Dataset date range: {returns_df.index.min().date()} .. {returns_df.index.max().date()}",
            f"- OOS returns shape: {oos.shape}",
            f"- weights_history shape: {weights_df.shape}",
            f"- risk_contributions shape: {rc_df.shape}",
            f"- diversification_ratio shape: {dr_df.shape}",
            f"- ENB shape: {enb_df.shape}",
            "",
            "## Sanity checks",
            "### Passed",
        ]
    )
    lines.extend([f"- {x}" for x in checks["passed"]])
    lines.append("### Failed")
    if checks["failed"]:
        lines.extend([f"- {x}" for x in checks["failed"]])
    else:
        lines.append("- None")

    lines.extend(["", "## Top 5 strategies by Sharpe"])
    lines.extend([f"- {k}: {v:.6f}" for k, v in top_sharpe.items()])
    lines.extend(["", "## Top 5 strategies by CVaR (higher is better / less negative)"])
    lines.extend([f"- {k}: {v:.6f}" for k, v in top_cvar.items()])
    lines.extend(["", "## Top 5 strategies by DR"])
    lines.extend([f"- {k}: {v:.6f}" for k, v in top_dr.items()])
    lines.extend(["", "## Top 5 strategies by ENB"])
    lines.extend([f"- {k}: {v:.6f}" for k, v in top_enb.items()])

    lines.extend(
        [
            "",
            "## Key conclusions",
            "- Crypto effect: generally increases CAGR and diversification for many families, but often increases volatility, CVaR loss, and drawdown.",
            "- RUONIA dominance: visible in defensive strategies; explicit RUONIA cap diagnostics are now exportable via weights and RUONIA summaries.",
            "- MV behavior: constrained MV remains very defensive; sample-MV can collapse toward low-volatility assets.",
            "- Hybrid behavior: variants are distinct; RUONIA-capped variants allow explicit cap-binding diagnostics.",
            "- Final candidate strategies (shortlist, not final selection): `rp_shrink_base`, `rp_shrink_crypto`, `aw_crypto`, `hybrid_ruonia_capped_crypto`, `mv_shrink_crypto`.",
            "",
            "## ENB note",
            "ENB is implemented as a risk-contribution-based ENB proxy: ENB_RC_Proxy = 1 / sum(rc_share_i^2).",
            "",
            "## Sortino note",
            "Sortino is undefined when downside deviation equals zero; exported thesis tables display it as N/A.",
        ]
    )
    VALIDATION_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    returns_df = load_returns_dataset(CSV_PATH)
    specs = _strategy_specs()
    weights_df, rc_df = _compute_weights_and_rc(returns_df, specs)
    weights_df.to_csv(WEIGHTS_HISTORY_CSV, index=False)
    rc_df.to_csv(RISK_CONTRIB_CSV, index=False)

    checks = _sanity_checks(weights_df, rc_df)

    crypto_summary, ruonia_summary = _save_summaries(weights_df, rc_df)
    enb_summary = _save_enb(rc_df)

    oos = pd.read_csv(OOS_RETURNS_CSV, index_col=0, parse_dates=True).sort_index()
    metrics_full = pd.read_csv(METRICS_FULL_CSV, index_col=0)
    metrics_crisis = pd.read_csv(METRICS_CRISIS_CSV, index_col=0)
    metrics_non = pd.read_csv(METRICS_NON_CRISIS_CSV, index_col=0)
    dr_df = pd.read_csv(DIVERSIFICATION_RATIO_CSV, parse_dates=["rebalance_date"])
    enb_df = pd.read_csv(ENB_CSV, parse_dates=["rebalance_date"])

    _save_tables(
        metrics_full=metrics_full,
        metrics_crisis=metrics_crisis,
        metrics_non_crisis=metrics_non,
        weights_df=weights_df,
        crypto_summary=crypto_summary,
        ruonia_summary=ruonia_summary,
        rc_summary=pd.read_csv(RISK_CONTRIB_SUMMARY_CSV),
        enb_summary=enb_summary,
    )
    _save_figures(oos, weights_df, rc_df, metrics_full, metrics_crisis, metrics_non, enb_summary)
    _write_validation_report(
        returns_df=returns_df,
        oos=oos,
        weights_df=weights_df,
        rc_df=rc_df,
        dr_df=dr_df,
        enb_df=enb_df,
        metrics_full=metrics_full,
        checks=checks,
    )

    print(f"Saved: {WEIGHTS_HISTORY_CSV}")
    print(f"Saved: {RISK_CONTRIB_CSV}")
    print(f"Saved: {WEIGHTS_SUMMARY_CSV}")
    print(f"Saved: {CRYPTO_EXPOSURE_SUMMARY_CSV}")
    print(f"Saved: {RUONIA_EXPOSURE_SUMMARY_CSV}")
    print(f"Saved: {RISK_CONTRIB_SUMMARY_CSV}")
    print(f"Saved: {CRYPTO_RISK_CONTRIB_SUMMARY_CSV}")
    print(f"Saved: {RUONIA_RISK_CONTRIB_SUMMARY_CSV}")
    print(f"Saved: {ENB_CSV}")
    print(f"Saved: {ENB_SUMMARY_CSV}")
    print(f"Saved tables dir: {TABLES_DIR}")
    print(f"Saved figures dir: {FIGURES_DIR}")
    print(f"Saved report: {VALIDATION_REPORT_MD}")
    if checks["failed"]:
        print("FAILED CHECKS:")
        for x in checks["failed"]:
            print(f"- {x}")
    else:
        print("All sanity checks passed.")


if __name__ == "__main__":
    main()

