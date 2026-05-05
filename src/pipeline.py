"""
Run full OOS backtest suite and write primary metrics tables to results/.
"""
import itertools
import numpy as np
import pandas as pd
from metrics import compute_all_metrics
from paths import (
    DIVERSIFICATION_RATIO_CSV,
    HYBRID_DIAGNOSTICS_CSV,
    METRICS_CSV,
    OOS_RETURNS_CSV,
)
from riskparity import (
    CSV_PATH,
    ROLLING_WINDOW_WEEKS,
    REBALANCE_EVERY_WEEKS,
    BASE_ASSETS,
    CRYPTO_ASSETS,
    load_returns_dataset,
    rolling_equal_weight_backtest,
    rolling_min_variance_backtest,
    rolling_risk_parity_backtest,
    compute_min_variance_weights,
    compute_risk_parity_weights,
)
from allweather import (
    BASE_WEIGHTS,
    CRYPTO_WEIGHTS,
    rolling_all_weather_backtest,
)
from rpallw import (
    BASE_CAPS,
    CRYPTO_CAPS,
    compute_hybrid_weights,
    hybrid_aw_rp_constrained,
    hybrid_aw_rp_loose,
    hybrid_aw_rp_unconstrained,
    hybrid_aw_rp_ruonia_capped,
    print_hybrid_weight_comparison,
)


def _estimate_covariance(train_slice: pd.DataFrame, cov_type: str) -> np.ndarray:
    if cov_type == "sample":
        return train_slice.cov().to_numpy()
    if cov_type == "shrinkage":
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf()
        lw.fit(train_slice.to_numpy())
        return lw.covariance_
    raise ValueError(f"Unsupported cov_type: {cov_type}")


def _compute_diversification_ratio(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    sigmas = np.sqrt(np.diag(cov_matrix))
    numer = float(weights @ sigmas)
    denom_sq = float(weights @ cov_matrix @ weights)
    if denom_sq <= 0.0:
        raise ValueError("Diversification ratio denominator must be positive.")
    dr = numer / np.sqrt(denom_sq)
    if not np.isfinite(dr):
        raise ValueError("Diversification ratio is not finite.")
    return float(dr)


def _rebalance_dates(data: pd.DataFrame, window: int, step: int) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    start = window
    n_obs = len(data)
    while start < n_obs:
        dates.append(data.index[start])
        start += step
    return dates


def build_diversification_ratio_panel(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_rows(strategy: str, assets: list[str], weight_getter, cov_type: str) -> None:
        data = returns_df.loc[:, assets].dropna().copy()
        start = ROLLING_WINDOW_WEEKS
        n_obs = len(data)
        while start < n_obs:
            train_slice = data.iloc[start - ROLLING_WINDOW_WEEKS : start]
            if len(train_slice) != ROLLING_WINDOW_WEEKS:
                break
            cov = _estimate_covariance(train_slice, cov_type=cov_type)
            w = np.asarray(weight_getter(cov), dtype=float)
            dr = _compute_diversification_ratio(w, cov)
            rows.append(
                {
                    "rebalance_date": data.index[start],
                    "strategy": strategy,
                    "diversification_ratio": dr,
                }
            )
            start += REBALANCE_EVERY_WEEKS

    add_rows("ew_base", BASE_ASSETS, lambda cov: np.full(len(BASE_ASSETS), 1.0 / len(BASE_ASSETS)), "sample")
    add_rows(
        "ew_crypto",
        CRYPTO_ASSETS,
        lambda cov: np.full(len(CRYPTO_ASSETS), 1.0 / len(CRYPTO_ASSETS)),
        "sample",
    )
    add_rows("mv_sample_base", BASE_ASSETS, compute_min_variance_weights, "sample")
    add_rows("mv_sample_crypto", CRYPTO_ASSETS, compute_min_variance_weights, "sample")
    add_rows("mv_shrink_base", BASE_ASSETS, compute_min_variance_weights, "shrinkage")
    add_rows("mv_shrink_crypto", CRYPTO_ASSETS, compute_min_variance_weights, "shrinkage")
    add_rows("rp_sample_base", BASE_ASSETS, compute_risk_parity_weights, "sample")
    add_rows("rp_sample_crypto", CRYPTO_ASSETS, compute_risk_parity_weights, "sample")
    add_rows("rp_shrink_base", BASE_ASSETS, compute_risk_parity_weights, "shrinkage")
    add_rows("rp_shrink_crypto", CRYPTO_ASSETS, compute_risk_parity_weights, "shrinkage")
    add_rows(
        "aw_base",
        list(BASE_WEIGHTS.keys()),
        lambda cov: np.array([BASE_WEIGHTS[a] for a in BASE_WEIGHTS], dtype=float),
        "sample",
    )
    add_rows(
        "aw_crypto",
        list(CRYPTO_WEIGHTS.keys()),
        lambda cov: np.array([CRYPTO_WEIGHTS[a] for a in CRYPTO_WEIGHTS], dtype=float),
        "sample",
    )
    add_rows(
        "hybrid_base",
        list(BASE_CAPS.keys()),
        lambda cov: compute_hybrid_weights(cov, BASE_CAPS, regime="constrained")[1],
        "sample",
    )
    add_rows(
        "hybrid_crypto",
        list(CRYPTO_CAPS.keys()),
        lambda cov: compute_hybrid_weights(cov, CRYPTO_CAPS, regime="constrained")[1],
        "sample",
    )
    add_rows(
        "hybrid_loose_base",
        list(BASE_CAPS.keys()),
        lambda cov: compute_hybrid_weights(cov, BASE_CAPS, regime="loose")[1],
        "sample",
    )
    add_rows(
        "hybrid_loose_crypto",
        list(CRYPTO_CAPS.keys()),
        lambda cov: compute_hybrid_weights(cov, CRYPTO_CAPS, regime="loose")[1],
        "sample",
    )
    add_rows(
        "hybrid_unconstrained_base",
        list(BASE_CAPS.keys()),
        lambda cov: compute_hybrid_weights(cov, BASE_CAPS, regime="unconstrained")[1],
        "sample",
    )
    add_rows(
        "hybrid_unconstrained_crypto",
        list(CRYPTO_CAPS.keys()),
        lambda cov: compute_hybrid_weights(cov, CRYPTO_CAPS, regime="unconstrained")[1],
        "sample",
    )
    add_rows(
        "hybrid_ruonia_capped_base",
        list(BASE_CAPS.keys()),
        lambda cov: compute_hybrid_weights(cov, BASE_CAPS, regime="constrained", ruonia_cap=0.25)[1],
        "sample",
    )
    add_rows(
        "hybrid_ruonia_capped_crypto",
        list(CRYPTO_CAPS.keys()),
        lambda cov: compute_hybrid_weights(cov, CRYPTO_CAPS, regime="constrained", ruonia_cap=0.25)[1],
        "sample",
    )
    dr_df = pd.DataFrame(rows).sort_values(["rebalance_date", "strategy"]).reset_index(drop=True)
    return dr_df


def _hybrid_diagnostics(oos_returns_df: pd.DataFrame) -> pd.DataFrame:
    hybrid_cols = [c for c in oos_returns_df.columns if "hybrid" in c]
    rows = []
    for left, right in itertools.combinations(hybrid_cols, 2):
        diff = oos_returns_df[left] - oos_returns_df[right]
        rows.append(
            {
                "left": left,
                "right": right,
                "max_abs_diff": float(diff.abs().max()),
                "correlation": float(oos_returns_df[left].corr(oos_returns_df[right])),
                "exact_equal": bool((oos_returns_df[left] == oos_returns_df[right]).all()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["exact_equal", "max_abs_diff"], ascending=[True, False])
    return out


def run_all_strategies(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all strategy variants (base and crypto) and collect OOS returns.
    """
    series_map = {
        "ew_base": rolling_equal_weight_backtest(
            returns=returns_df,
            assets=BASE_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "ew_crypto": rolling_equal_weight_backtest(
            returns=returns_df,
            assets=CRYPTO_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "mv_sample_base": rolling_min_variance_backtest(
            returns=returns_df,
            assets=BASE_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
            cov_type="sample",
        ),
        "mv_sample_crypto": rolling_min_variance_backtest(
            returns=returns_df,
            assets=CRYPTO_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
            cov_type="sample",
        ),
        "mv_shrink_base": rolling_min_variance_backtest(
            returns=returns_df,
            assets=BASE_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
            cov_type="shrinkage",
        ),
        "mv_shrink_crypto": rolling_min_variance_backtest(
            returns=returns_df,
            assets=CRYPTO_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
            cov_type="shrinkage",
        ),
        "rp_sample_base": rolling_risk_parity_backtest(
            returns=returns_df,
            assets=BASE_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
            cov_type="sample",
        ),
        "rp_sample_crypto": rolling_risk_parity_backtest(
            returns=returns_df,
            assets=CRYPTO_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
            cov_type="sample",
        ),
        "rp_shrink_base": rolling_risk_parity_backtest(
            returns=returns_df,
            assets=BASE_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
            cov_type="shrinkage",
        ),
        "rp_shrink_crypto": rolling_risk_parity_backtest(
            returns=returns_df,
            assets=CRYPTO_ASSETS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
            cov_type="shrinkage",
        ),
        "aw_base": rolling_all_weather_backtest(
            returns=returns_df,
            weights_dict=BASE_WEIGHTS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "aw_crypto": rolling_all_weather_backtest(
            returns=returns_df,
            weights_dict=CRYPTO_WEIGHTS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "hybrid_base": hybrid_aw_rp_constrained(
            returns=returns_df,
            caps_dict=BASE_CAPS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "hybrid_crypto": hybrid_aw_rp_constrained(
            returns=returns_df,
            caps_dict=CRYPTO_CAPS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "hybrid_loose_base": hybrid_aw_rp_loose(
            returns=returns_df,
            caps_dict=BASE_CAPS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "hybrid_loose_crypto": hybrid_aw_rp_loose(
            returns=returns_df,
            caps_dict=CRYPTO_CAPS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "hybrid_unconstrained_base": hybrid_aw_rp_unconstrained(
            returns=returns_df,
            caps_dict=BASE_CAPS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "hybrid_unconstrained_crypto": hybrid_aw_rp_unconstrained(
            returns=returns_df,
            caps_dict=CRYPTO_CAPS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "hybrid_ruonia_capped_base": hybrid_aw_rp_ruonia_capped(
            returns=returns_df,
            caps_dict=BASE_CAPS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
        "hybrid_ruonia_capped_crypto": hybrid_aw_rp_ruonia_capped(
            returns=returns_df,
            caps_dict=CRYPTO_CAPS,
            window=ROLLING_WINDOW_WEEKS,
            step=REBALANCE_EVERY_WEEKS,
        ),
    }

    for name, s in series_map.items():
        s.name = name

    oos_returns_df = pd.concat(series_map.values(), axis=1)
    oos_returns_df = oos_returns_df.sort_index().dropna(how="any")
    return oos_returns_df


def run_pipeline() -> None:
    returns_df = load_returns_dataset(CSV_PATH)
    print_hybrid_weight_comparison(
        returns=returns_df,
        caps_dict=CRYPTO_CAPS,
        label="CRYPTO",
        window=ROLLING_WINDOW_WEEKS,
        steps_to_print=2,
    )
    df = run_all_strategies(returns_df)
    df.to_csv(OOS_RETURNS_CSV, index=True)
    hybrid_diag = _hybrid_diagnostics(df)
    hybrid_diag.to_csv(HYBRID_DIAGNOSTICS_CSV, index=False)

    dr_df = build_diversification_ratio_panel(returns_df)
    dr_df.to_csv(DIVERSIFICATION_RATIO_CSV, index=False)

    metrics_df = compute_all_metrics(df)
    metrics_df.to_csv(METRICS_CSV, index=True)

    print(f"Saved OOS returns: {OOS_RETURNS_CSV}")
    print(f"Saved diversification ratios: {DIVERSIFICATION_RATIO_CSV}")
    print(f"Saved hybrid diagnostics: {HYBRID_DIAGNOSTICS_CSV}")
    print(f"Saved metrics: {METRICS_CSV}")
    print(f"Shape: {df.shape}")
