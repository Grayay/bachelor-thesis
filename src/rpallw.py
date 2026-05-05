import numpy as np
import pandas as pd

from riskparity import (
    CSV_PATH,
    ROLLING_WINDOW_WEEKS,
    REBALANCE_EVERY_WEEKS,
    load_returns_dataset,
    compute_risk_parity_weights,
    _portfolio_returns_from_asset_log_returns,
    validate_and_print_stats,
    plot_cumulative_log_returns,
)

# -----------------------------
# Hybrid Constraints (caps)
# -----------------------------
BASE_CAPS = {
    "moex_return": 0.30,
    "ofz_return": 0.35,
    "gold_return": 0.20,
    "ruonia_return": 0.15,
}

CRYPTO_CAPS = {
    "moex_return": 0.25,
    "ofz_return": 0.35,
    "gold_return": 0.20,
    "ruonia_return": 0.10,
    "btc_return": 0.05,
    "eth_return": 0.05,
}

HYBRID_ALPHA_CONSTRAINED = 0.5
HYBRID_ALPHA_LOOSE = 0.2
HYBRID_ALPHA_UNCONSTRAINED = 0.0


def _apply_ruonia_cap(
    weights: np.ndarray, assets: list[str], ruonia_cap: float = 0.25
) -> np.ndarray:
    """
    Cap RUONIA weight and redistribute excess across other positive-weight assets.
    """
    w = np.array(weights, dtype=float).copy()
    ruonia_idx = assets.index("ruonia_return")
    if w[ruonia_idx] <= ruonia_cap:
        return w / w.sum()

    excess = w[ruonia_idx] - ruonia_cap
    w[ruonia_idx] = ruonia_cap

    other_idx = [i for i in range(len(w)) if i != ruonia_idx and w[i] > 0.0]
    if not other_idx:
        raise ValueError("Cannot redistribute RUONIA excess: no positive other weights.")

    other_weights = w[other_idx]
    other_sum = other_weights.sum()
    if np.isclose(other_sum, 0.0):
        raise ValueError("Cannot redistribute RUONIA excess: zero weight pool.")

    w[other_idx] = other_weights + excess * (other_weights / other_sum)
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def compute_hybrid_weights(
    cov_matrix: np.ndarray,
    caps_dict: dict[str, float],
    regime: str,
    ruonia_cap: float | None = None,
) -> tuple[list[str], np.ndarray]:
    assets = list(caps_dict.keys())
    aw_weights = np.array([caps_dict[a] for a in assets], dtype=float)
    # For constrained/loose hybrids we keep AW-anchored initialization.
    # For unconstrained (alpha=0 by default), we intentionally align with the
    # pure RP path used elsewhere in the project (default RP initialization).
    if regime == "unconstrained":
        rp_weights = compute_risk_parity_weights(cov_matrix)
    else:
        rp_weights = compute_risk_parity_weights(cov_matrix, x0=aw_weights)

    if regime == "constrained":
        alpha = HYBRID_ALPHA_CONSTRAINED
        hybrid_weights = alpha * aw_weights + (1.0 - alpha) * rp_weights
        hybrid_weights = np.clip(hybrid_weights, 0.0, None)
        hybrid_weights = hybrid_weights / hybrid_weights.sum()
    elif regime == "loose":
        alpha = HYBRID_ALPHA_LOOSE
        hybrid_weights = alpha * aw_weights + (1.0 - alpha) * rp_weights
        hybrid_weights = np.clip(hybrid_weights, 0.0, None)
        hybrid_weights = hybrid_weights / hybrid_weights.sum()
    elif regime == "unconstrained":
        alpha = HYBRID_ALPHA_UNCONSTRAINED
        hybrid_weights = alpha * aw_weights + (1.0 - alpha) * rp_weights
        hybrid_weights = np.clip(hybrid_weights, 0.0, None)
        hybrid_weights = hybrid_weights / hybrid_weights.sum()
    else:
        raise ValueError(f"Unknown hybrid regime: {regime}")

    if ruonia_cap is not None:
        hybrid_weights = _apply_ruonia_cap(hybrid_weights, assets, ruonia_cap=ruonia_cap)

    return assets, hybrid_weights


def _rolling_hybrid_generic(
    returns: pd.DataFrame,
    caps_dict: dict[str, float],
    regime: str,
    alpha_override: float | None = None,
    ruonia_cap: float | None = None,
    window: int = 156,
    step: int = 13,
    output: str = "log",
) -> pd.Series:
    """
    Rolling out-of-sample backtest for All-Weather + Risk Parity hybrid:
      - compute RP (ERC) weights on training window
      - apply All-Weather caps
      - renormalize
      - apply fixed adjusted weights to next test block
    """
    assets = list(caps_dict.keys())
    data = returns.loc[:, assets].dropna().copy()

    if len(data) <= window:
        raise ValueError(
            f"Not enough observations for backtest: have {len(data)}, need > {window}."
        )

    oos_blocks = []
    start = window
    n_obs = len(data)

    while start < n_obs:
        train_slice = data.iloc[start - window : start]
        if len(train_slice) != window:
            break

        cov_matrix = train_slice.cov().to_numpy()
        _, hybrid_weights = compute_hybrid_weights(
            cov_matrix=cov_matrix,
            caps_dict=caps_dict,
            regime=regime,
            ruonia_cap=ruonia_cap,
        )
        if alpha_override is not None:
            aw_weights = np.array([caps_dict[a] for a in assets], dtype=float)
            rp_weights = compute_risk_parity_weights(cov_matrix, x0=aw_weights)
            hybrid_weights = alpha_override * aw_weights + (1.0 - alpha_override) * rp_weights
            hybrid_weights = np.clip(hybrid_weights, 0.0, None)
            hybrid_weights = hybrid_weights / hybrid_weights.sum()

        end = min(start + step, n_obs)
        test_slice = data.iloc[start:end]
        if test_slice.empty:
            break

        portfolio_test_returns = _portfolio_returns_from_asset_log_returns(
            test_slice, hybrid_weights, output=output
        )
        block_series = pd.Series(
            portfolio_test_returns,
            index=test_slice.index,
            name="hybrid_returns",
        )
        oos_blocks.append(block_series)
        start += step

    if not oos_blocks:
        raise ValueError("Backtest produced no out-of-sample data.")

    oos = pd.concat(oos_blocks).sort_index()
    if oos.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in OOS output.")
    return oos


def hybrid_aw_rp_constrained(
    returns: pd.DataFrame,
    caps_dict: dict[str, float],
    alpha: float = HYBRID_ALPHA_CONSTRAINED,
    window: int = 156,
    step: int = 13,
    output: str = "log",
) -> pd.Series:
    """
    Existing hybrid logic kept under explicit constrained function name.
    """
    return _rolling_hybrid_generic(
        returns=returns,
        caps_dict=caps_dict,
        regime="constrained",
        alpha_override=alpha if not np.isclose(alpha, HYBRID_ALPHA_CONSTRAINED) else None,
        window=window,
        step=step,
        output=output,
    )


def hybrid_aw_rp_loose(
    returns: pd.DataFrame,
    caps_dict: dict[str, float],
    alpha: float = HYBRID_ALPHA_LOOSE,
    window: int = 156,
    step: int = 13,
    output: str = "log",
) -> pd.Series:
    """
    Hybrid with loose crypto bounds: BTC/ETH <= 0.3, others <= 1.
    """
    return _rolling_hybrid_generic(
        returns=returns,
        caps_dict=caps_dict,
        regime="loose",
        alpha_override=alpha if not np.isclose(alpha, HYBRID_ALPHA_LOOSE) else None,
        window=window,
        step=step,
        output=output,
    )


def hybrid_aw_rp_unconstrained(
    returns: pd.DataFrame,
    caps_dict: dict[str, float],
    alpha: float = HYBRID_ALPHA_UNCONSTRAINED,
    window: int = 156,
    step: int = 13,
    output: str = "log",
) -> pd.Series:
    """
    Hybrid with non-negativity only and (0,1) bounds for all assets.
    """
    return _rolling_hybrid_generic(
        returns=returns,
        caps_dict=caps_dict,
        regime="unconstrained",
        alpha_override=alpha if not np.isclose(alpha, HYBRID_ALPHA_UNCONSTRAINED) else None,
        window=window,
        step=step,
        output=output,
    )


def hybrid_aw_rp_ruonia_capped(
    returns: pd.DataFrame,
    caps_dict: dict[str, float],
    alpha: float = HYBRID_ALPHA_CONSTRAINED,
    ruonia_cap: float = 0.25,
    window: int = 156,
    step: int = 13,
    output: str = "log",
) -> pd.Series:
    """
    Hybrid portfolio with post-processing RUONIA cap.
    """
    return _rolling_hybrid_generic(
        returns=returns,
        caps_dict=caps_dict,
        regime="constrained",
        alpha_override=alpha if not np.isclose(alpha, HYBRID_ALPHA_CONSTRAINED) else None,
        ruonia_cap=ruonia_cap,
        window=window,
        step=step,
        output=output,
    )


# Backward-compatible alias.
def rolling_hybrid_backtest(
    returns: pd.DataFrame,
    caps_dict: dict[str, float],
    alpha: float = HYBRID_ALPHA_CONSTRAINED,
    window: int = 156,
    step: int = 13,
    output: str = "log",
) -> pd.Series:
    return hybrid_aw_rp_constrained(
        returns=returns,
        caps_dict=caps_dict,
        alpha=alpha,
        window=window,
        step=step,
        output=output,
    )


def print_hybrid_weight_comparison(
    returns: pd.DataFrame,
    caps_dict: dict[str, float],
    label: str,
    window: int = 156,
    steps_to_print: int = 2,
) -> None:
    assets = list(caps_dict.keys())
    data = returns.loc[:, assets].dropna().copy()
    for i in range(steps_to_print):
        start = window + i * REBALANCE_EVERY_WEEKS
        train_slice = data.iloc[start - window : start]
        cov_matrix = train_slice.cov().to_numpy()
        constrained = compute_hybrid_weights(cov_matrix, caps_dict, regime="constrained")[1]
        loose = compute_hybrid_weights(cov_matrix, caps_dict, regime="loose")[1]
        unconstrained = compute_hybrid_weights(cov_matrix, caps_dict, regime="unconstrained")[1]
        capped = compute_hybrid_weights(
            cov_matrix, caps_dict, regime="constrained", ruonia_cap=0.25
        )[1]
        ruonia_idx = assets.index("ruonia_return")

        print(f"[{label}] Rebalance step {i + 1}")
        print("hybrid_constrained weights:", constrained)
        print("hybrid_loose weights:", loose)
        print("hybrid_unconstrained weights:", unconstrained)
        print("hybrid_ruonia_capped weights:", capped)
        print("RUONIA weight (capped):", capped[ruonia_idx])
        print(
            "different:",
            not np.allclose(constrained, loose, atol=1e-6, rtol=1e-5)
            or not np.allclose(constrained, unconstrained, atol=1e-6, rtol=1e-5)
            or not np.allclose(loose, unconstrained, atol=1e-6, rtol=1e-5),
        )


if __name__ == "__main__":
    returns_df = load_returns_dataset(CSV_PATH)

    hybrid_base_returns = hybrid_aw_rp_constrained(
        returns=returns_df,
        caps_dict=BASE_CAPS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
    )
    hybrid_crypto_returns = hybrid_aw_rp_constrained(
        returns=returns_df,
        caps_dict=CRYPTO_CAPS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
    )

    validate_and_print_stats("hybrid_base_returns", hybrid_base_returns)
    validate_and_print_stats("hybrid_crypto_returns", hybrid_crypto_returns)

    plot_cumulative_log_returns(
        hybrid_base_returns,
        hybrid_crypto_returns,
        base_label="Hybrid Base",
        crypto_label="Hybrid Crypto",
        title="All-Weather + Risk Parity: Base vs Crypto",
    )
