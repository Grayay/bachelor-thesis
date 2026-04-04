"""
Run full OOS backtest suite and write primary metrics tables to results/.
"""
import pandas as pd
from metrics import compute_all_metrics
from paths import METRICS_CSV, OOS_RETURNS_CSV
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
)
from allweather import (
    BASE_WEIGHTS,
    CRYPTO_WEIGHTS,
    rolling_all_weather_backtest,
)
from rpallw import (
    BASE_CAPS,
    CRYPTO_CAPS,
    hybrid_aw_rp_constrained,
    hybrid_aw_rp_loose,
    hybrid_aw_rp_unconstrained,
    hybrid_aw_rp_ruonia_capped,
    print_hybrid_weight_comparison,
)


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

    metrics_df = compute_all_metrics(df)
    metrics_df.to_csv(METRICS_CSV, index=True)

    print(f"Saved OOS returns: {OOS_RETURNS_CSV}")
    print(f"Saved metrics: {METRICS_CSV}")
    print(f"Shape: {df.shape}")
