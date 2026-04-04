import numpy as np

from riskparity import (
    CSV_PATH,
    ROLLING_WINDOW_WEEKS,
    CRYPTO_ASSETS,
    load_returns_dataset,
    compute_risk_parity_weights,
)


def get_one_real_covariance() -> np.ndarray:
    """
    Use the same rolling-window logic as backtests, but only first window.
    """
    data = load_returns_dataset(CSV_PATH).loc[:, CRYPTO_ASSETS].dropna().copy()
    start = ROLLING_WINDOW_WEEKS
    train_slice = data.iloc[start - ROLLING_WINDOW_WEEKS : start]
    return train_slice.cov().to_numpy()


def inverse_volatility_weights(cov_matrix: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vol = 1.0 / vols
    return inv_vol / inv_vol.sum()


def portfolio_diagnostics(cov_matrix: np.ndarray, w: np.ndarray):
    portfolio_var = float(w @ cov_matrix @ w)
    marginal = cov_matrix @ w
    rc = w * marginal
    rc_shares = rc / rc.sum()
    return portfolio_var, marginal, rc, rc_shares


if __name__ == "__main__":
    cov = get_one_real_covariance()
    n = cov.shape[0]

    w_ew = np.full(n, 1.0 / n)
    w_iv = inverse_volatility_weights(cov)
    w_rp = compute_risk_parity_weights(cov)

    names = ["Equal Weight", "Inverse Volatility", "Risk Parity"]
    weights_list = [w_ew, w_iv, w_rp]

    dispersion = {}
    for name, w in zip(names, weights_list):
        var, _, rc, rc_shares = portfolio_diagnostics(cov, w)
        std_rc_shares = float(np.std(rc_shares))
        range_rc_shares = float(np.max(rc_shares) - np.min(rc_shares))
        dispersion[name] = (std_rc_shares, range_rc_shares)

        print(f"=== {name} ===")
        print("weights:", w)
        print("portfolio variance:", var)
        print("RC:", rc)
        print("RC shares:", rc_shares)
        print("RC share std:", std_rc_shares)
        print("RC share max-min:", range_rc_shares)
        print()

    best = min(dispersion, key=lambda k: dispersion[k][0])
    print("Conclusion:")
    print(f"Most equalized risk contributions (lowest RC-share std): {best}")
