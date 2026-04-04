import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from riskparity import (
    CSV_PATH,
    ROLLING_WINDOW_WEEKS,
    load_returns_dataset,
    compute_risk_parity_weights,
)
from rpallw import CRYPTO_CAPS


def compute_one_step_hybrid_weights(aw_weights: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Compute hybrid weights for the first rebalance step only.
    """
    assets = list(CRYPTO_CAPS.keys())
    data = load_returns_dataset(CSV_PATH).loc[:, assets].dropna().copy()

    start = ROLLING_WINDOW_WEEKS
    train_slice = data.iloc[start - ROLLING_WINDOW_WEEKS : start]
    cov_matrix = train_slice.cov().to_numpy()

    rp_weights = compute_risk_parity_weights(cov_matrix, x0=aw_weights)
    hybrid = alpha * aw_weights + (1.0 - alpha) * rp_weights
    return hybrid / hybrid.sum()


if __name__ == "__main__":
    # BASE (original)
    aw_base = np.array([0.25, 0.35, 0.20, 0.10, 0.05, 0.05], dtype=float)

    # MODIFIED: reduce crypto, redistribute equally to non-crypto assets
    aw_modified = np.array([0.2625, 0.3625, 0.2125, 0.1125, 0.025, 0.025], dtype=float)

    hybrid_base = compute_one_step_hybrid_weights(aw_base)
    hybrid_modified = compute_one_step_hybrid_weights(aw_modified)

    print("BASE AW weights:", aw_base)
    print("BASE Hybrid weights:", hybrid_base)
    print("BASE (Hybrid - AW):", hybrid_base - aw_base)
    print()
    print("MODIFIED AW weights:", aw_modified)
    print("MODIFIED Hybrid weights:", hybrid_modified)
    print("MODIFIED (Hybrid - AW):", hybrid_modified - aw_modified)
