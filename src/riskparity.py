import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from paths import FINAL_DATASET_CSV

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = str(FINAL_DATASET_CSV)
ROLLING_WINDOW_WEEKS = 156
REBALANCE_EVERY_WEEKS = 13
BASE_ASSETS = ["moex_return", "ofz_return", "gold_return", "ruonia_return"]
CRYPTO_ASSETS = [
    "moex_return",
    "ofz_return",
    "gold_return",
    "ruonia_return",
    "btc_return",
    "eth_return",
]


# -----------------------------
# Data Loading
# -----------------------------
def load_returns_dataset(path: str) -> pd.DataFrame:
    """
    Load weekly log returns dataset from CSV, parse date, set as index.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def _portfolio_returns_from_asset_log_returns(
    asset_log_returns: pd.DataFrame, weights: np.ndarray, output: str = "log"
) -> np.ndarray:
    """
    Convert asset log returns -> portfolio return using fixed weights.

    For asset log returns r_{i,t}:
      R_{i,t} = exp(r_{i,t}) - 1
      R_{p,t} = sum_i w_i * R_{i,t}
      r_{p,t} = log1p(R_{p,t})
    """
    if output not in {"log", "simple"}:
        raise ValueError("output must be either 'log' or 'simple'.")
    asset_simple = np.expm1(asset_log_returns.to_numpy())
    port_simple = asset_simple @ weights
    return np.log1p(port_simple) if output == "log" else port_simple


# -----------------------------
# Backtest Logic
# -----------------------------
def rolling_equal_weight_backtest(
    returns: pd.DataFrame,
    assets: list[str],
    window: int = 156,
    step: int = 13,
    output: str = "log",
) -> pd.Series:
    """
    Rolling out-of-sample backtest for an Equal Weight (1/N) portfolio.
    At each iteration:
      - train = past `window` observations
      - weights = equal weights across assets
      - test = next `step` observations
      - apply fixed weights over test block
      - move forward by `step`
    Returns:
      OOS portfolio log return series (pd.Series, datetime index).
    """
    data = returns.loc[:, assets].dropna().copy()
    if len(data) <= window:
        raise ValueError(
            f"Not enough observations for backtest: have {len(data)}, need > {window}."
        )
    oos_blocks = []
    n_assets = len(assets)
    weights = np.full(n_assets, 1.0 / n_assets)
    start = window
    n_obs = len(data)
    while start < n_obs:
        # a) training window (kept for correct rolling structure / validation)
        train_slice = data.iloc[start - window : start]
        if len(train_slice) != window:
            break
        # b) equal weights already defined (1/N)
        # c) apply weights to next test window
        end = min(start + step, n_obs)
        test_slice = data.iloc[start:end]
        if test_slice.empty:
            break
        portfolio_test_returns = _portfolio_returns_from_asset_log_returns(
            test_slice, weights, output=output
        )
        block_series = pd.Series(
            portfolio_test_returns,
            index=test_slice.index,
            name="ew_returns",
        )
        oos_blocks.append(block_series)
        # move forward by rebalance interval
        start += step
    if not oos_blocks:
        raise ValueError("Backtest produced no out-of-sample data.")
    oos = pd.concat(oos_blocks).sort_index()
    # Guard against accidental duplicate dates (should not happen with non-overlapping windows)
    if oos.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in OOS output.")
    return oos


def compute_min_variance_weights(cov_matrix) -> np.ndarray:
    """
    Compute exact long-only minimum variance weights with SLSQP:
      minimize w' * Sigma * w
      subject to sum(w)=1, w_i>=0.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n_assets = cov.shape[0]

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")
    cov_reg = cov + 1e-8 * np.eye(n_assets)
    x0 = np.full(n_assets, 1.0 / n_assets)

    def objective(w: np.ndarray, cov_in: np.ndarray) -> float:
        return float(w.T @ cov_in @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_assets
    result = minimize(
        objective,
        x0,
        args=(cov_reg,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    if not result.success:
        raise RuntimeError(f"Minimum variance optimization failed: {result.message}")

    weights = np.asarray(result.x, dtype=float)
    tol = 1e-8
    if not np.isfinite(weights).all():
        raise RuntimeError("Minimum variance optimization returned non-finite weights.")
    if abs(weights.sum() - 1.0) > 1e-6:
        raise RuntimeError("Minimum variance weights do not satisfy sum(w)=1.")
    if (weights < -tol).any():
        raise RuntimeError("Minimum variance weights violate non-negativity.")

    weights = np.clip(weights, 0.0, 1.0)
    weights = weights / weights.sum()
    obj = objective(weights, cov_reg)
    if not np.isfinite(obj):
        raise RuntimeError("Minimum variance objective is not finite.")
    return weights


def rolling_min_variance_backtest(
    returns: pd.DataFrame,
    assets: list[str],
    window: int = 156,
    step: int = 13,
    cov_type: str = "sample",
    output: str = "log",
) -> pd.Series:
    """
    Rolling out-of-sample backtest for long-only Minimum Variance portfolio.

    At each iteration:
      - train = past `window` observations
      - compute sample covariance from train
      - compute long-only min-variance weights
      - apply fixed weights to next `step` observations
      - move forward by `step`
    """
    if cov_type not in {"sample", "shrinkage"}:
        raise ValueError("cov_type must be either 'sample' or 'shrinkage'.")

    data = returns.loc[:, assets].dropna().copy()
    if len(data) <= window:
        raise ValueError(
            f"Not enough observations for backtest: have {len(data)}, need > {window}."
        )

    oos_blocks = []
    start = window
    n_obs = len(data)

    while start < n_obs:
        # a) training window
        train_slice = data.iloc[start - window : start]
        if len(train_slice) != window:
            break

        # b) compute min variance weights
        if cov_type == "sample":
            cov_matrix = train_slice.cov().to_numpy()
        else:
            lw = LedoitWolf()
            lw.fit(train_slice.to_numpy())
            cov_matrix = lw.covariance_
        weights = compute_min_variance_weights(cov_matrix)

        # c) apply weights to next test window
        end = min(start + step, n_obs)
        test_slice = data.iloc[start:end]
        if test_slice.empty:
            break

        portfolio_test_returns = _portfolio_returns_from_asset_log_returns(
            test_slice, weights, output=output
        )
        block_series = pd.Series(
            portfolio_test_returns,
            index=test_slice.index,
            name="mv_returns",
        )
        oos_blocks.append(block_series)

        start += step

    if not oos_blocks:
        raise ValueError("Backtest produced no out-of-sample data.")

    oos = pd.concat(oos_blocks).sort_index()
    if oos.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in OOS output.")
    return oos


def compute_risk_parity_weights(cov_matrix, x0: np.ndarray | None = None) -> np.ndarray:
    """
    Compute long-only Equal Risk Contribution (ERC) weights.

    Constraints:
      - weights >= 0
      - sum(weights) = 1
    """
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")

    n_assets = cov.shape[0]
    cov_reg = cov + 1e-6 * np.eye(n_assets)
    if x0 is None:
        x0 = np.full(n_assets, 1.0 / n_assets)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape[0] != n_assets:
            raise ValueError("x0 length must match covariance dimension.")
        x0 = np.clip(x0, 0.0, None)
        x0_sum = x0.sum()
        if np.isclose(x0_sum, 0.0):
            x0 = np.full(n_assets, 1.0 / n_assets)
        else:
            x0 = x0 / x0_sum

    def rp_objective(w: np.ndarray, cov_in: np.ndarray) -> float:
        eps = 1e-12
        marginal = cov_in @ w
        rc = w * marginal
        rc_norm = rc / (rc.sum() + eps)
        target = np.ones_like(rc_norm) / len(w)
        return ((rc_norm - target) ** 2).sum()

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_assets
    result = minimize(
        rp_objective,
        x0,
        args=(cov_reg,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Risk parity optimization failed: {result.message}")

    weights = np.clip(result.x, 0.0, 1.0)
    weight_sum = weights.sum()
    if np.isclose(weight_sum, 0.0):
        raise RuntimeError("Risk parity optimization produced zero-sum weights.")

    weights = weights / weight_sum

    return weights


def rolling_risk_parity_backtest(
    returns: pd.DataFrame,
    assets: list[str],
    window: int = 156,
    step: int = 13,
    cov_type: str = "sample",
    output: str = "log",
) -> pd.Series:
    """
    Rolling out-of-sample backtest for long-only Risk Parity (ERC) portfolio.

    At each iteration:
      - train = past `window` observations
      - compute covariance from train (sample or Ledoit-Wolf shrinkage)
      - compute ERC weights
      - apply fixed weights to next `step` observations
      - move forward by `step`
    """
    if cov_type not in {"sample", "shrinkage"}:
        raise ValueError("cov_type must be either 'sample' or 'shrinkage'.")

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

        if cov_type == "sample":
            cov_matrix = train_slice.cov().to_numpy()
        else:
            lw = LedoitWolf()
            lw.fit(train_slice.to_numpy())
            cov_matrix = lw.covariance_

        weights = compute_risk_parity_weights(cov_matrix)

        end = min(start + step, n_obs)
        test_slice = data.iloc[start:end]
        if test_slice.empty:
            break

        portfolio_test_returns = _portfolio_returns_from_asset_log_returns(
            test_slice, weights, output=output
        )
        block_series = pd.Series(
            portfolio_test_returns,
            index=test_slice.index,
            name="rp_returns",
        )
        oos_blocks.append(block_series)

        start += step

    if not oos_blocks:
        raise ValueError("Backtest produced no out-of-sample data.")

    oos = pd.concat(oos_blocks).sort_index()
    if oos.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in OOS output.")
    return oos


# -----------------------------
# Validation & Reporting
# -----------------------------
def validate_and_print_stats(name: str, s: pd.Series) -> None:
    if s.isna().any():
        raise ValueError(f"{name} contains NaNs.")
    print(f"{name}:")
    print(f"  length = {len(s)}")
    print(f"  mean   = {s.mean():.8f}")
    print(f"  std    = {s.std(ddof=1):.8f}")
    print()


def plot_cumulative_log_returns(
    base: pd.Series,
    crypto: pd.Series,
    base_label: str = "Base",
    crypto_label: str = "Crypto",
    title: str = "Cumulative Returns",
) -> None:
    """
    For log returns r_t, cumulative wealth index is exp(cumsum(r_t)).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return

    base_cum = np.exp(base.cumsum())
    crypto_cum = np.exp(crypto.cumsum())
    plt.figure(figsize=(11, 6))
    plt.plot(base_cum.index, base_cum.values, label=base_label, linewidth=2)
    plt.plot(crypto_cum.index, crypto_cum.values, label=crypto_label, linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of 1 RUB")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    returns_df = load_returns_dataset(CSV_PATH)

    # Existing strategies (kept in project)
    ew_base_returns = rolling_equal_weight_backtest(
        returns=returns_df,
        assets=BASE_ASSETS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
    )
    ew_crypto_returns = rolling_equal_weight_backtest(
        returns=returns_df,
        assets=CRYPTO_ASSETS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
    )
    mv_base_returns = rolling_min_variance_backtest(
        returns=returns_df,
        assets=BASE_ASSETS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
    )
    mv_crypto_returns = rolling_min_variance_backtest(
        returns=returns_df,
        assets=CRYPTO_ASSETS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
    )

    # Risk parity (sample covariance)
    rp_base_sample_returns = rolling_risk_parity_backtest(
        returns=returns_df,
        assets=BASE_ASSETS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
        cov_type="sample",
    )
    rp_crypto_sample_returns = rolling_risk_parity_backtest(
        returns=returns_df,
        assets=CRYPTO_ASSETS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
        cov_type="sample",
    )

    # Risk parity (Ledoit-Wolf shrinkage covariance)
    rp_base_shrink_returns = rolling_risk_parity_backtest(
        returns=returns_df,
        assets=BASE_ASSETS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
        cov_type="shrinkage",
    )
    rp_crypto_shrink_returns = rolling_risk_parity_backtest(
        returns=returns_df,
        assets=CRYPTO_ASSETS,
        window=ROLLING_WINDOW_WEEKS,
        step=REBALANCE_EVERY_WEEKS,
        cov_type="shrinkage",
    )

    validate_and_print_stats("rp_base_sample_returns", rp_base_sample_returns)
    validate_and_print_stats("rp_crypto_sample_returns", rp_crypto_sample_returns)
    validate_and_print_stats("rp_base_shrink_returns", rp_base_shrink_returns)
    validate_and_print_stats("rp_crypto_shrink_returns", rp_crypto_shrink_returns)

    # Plot only crypto inclusion comparisons for risk parity
    plot_cumulative_log_returns(
        rp_base_sample_returns,
        rp_crypto_sample_returns,
        base_label="RP Sample Base",
        crypto_label="RP Sample Crypto",
        title="RP sample: Base vs Crypto",
    )
    plot_cumulative_log_returns(
        rp_base_shrink_returns,
        rp_crypto_shrink_returns,
        base_label="RP Shrinkage Base",
        crypto_label="RP Shrinkage Crypto",
        title="RP shrinkage: Base vs Crypto",
    )
