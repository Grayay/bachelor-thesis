import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


def _to_simple_returns(returns: pd.DataFrame, input_type: str) -> pd.DataFrame:
    if input_type not in {"log", "simple"}:
        raise ValueError("input_type must be 'log' or 'simple'.")
    if input_type == "simple":
        return returns
    # log -> simple
    return returns.apply(np.expm1)


def _log_sum_cagr(log_returns: pd.DataFrame) -> pd.Series:
    """
    CAGR from weekly log returns via total log return.
    """
    n = len(log_returns)
    total_log = log_returns.sum(axis=0)
    return np.expm1(total_log * (WEEKS_PER_YEAR / n))


def compute_cagr(returns: pd.DataFrame) -> pd.Series:
    """
    CAGR per strategy from weekly *simple* returns.
    """
    n = len(returns)
    return (1.0 + returns).prod(axis=0) ** (WEEKS_PER_YEAR / n) - 1.0


def compute_volatility(returns: pd.DataFrame) -> pd.Series:
    """
    Annualized volatility per strategy from weekly returns.
    """
    return returns.std(axis=0, ddof=1) * np.sqrt(WEEKS_PER_YEAR)


def compute_sharpe(returns: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """
    Annualized Sharpe ratio per strategy.
    """
    annual_return = returns.mean(axis=0) * WEEKS_PER_YEAR
    annual_vol = returns.std(axis=0, ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    return (annual_return - rf) / annual_vol


def compute_sortino(returns: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """
    Annualized Sortino ratio per strategy using downside deviation.
    """
    annual_return = returns.mean(axis=0) * WEEKS_PER_YEAR
    downside = returns.where(returns < 0.0, 0.0)
    downside_std = downside.std(axis=0, ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    return (annual_return - rf) / downside_std


def compute_max_drawdown(returns: pd.DataFrame) -> pd.Series:
    """
    Max drawdown per strategy from cumulative return paths (simple returns).
    """
    cumulative = (1.0 + returns).cumprod(axis=0)
    running_peak = cumulative.cummax(axis=0)
    drawdown = cumulative / running_peak - 1.0
    return drawdown.min(axis=0)


def compute_cvar(returns: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """
    CVaR (Expected Shortfall) per strategy at given alpha.
    """
    var = returns.quantile(alpha, axis=0)
    return returns.where(returns.le(var), np.nan).mean(axis=0)


def compute_skewness(returns: pd.DataFrame) -> pd.Series:
    return returns.skew(axis=0)


def compute_kurtosis(returns: pd.DataFrame) -> pd.Series:
    """
    Excess kurtosis (Fisher), consistent with pandas default.
    """
    return returns.kurt(axis=0)


def compute_diversification_ratio(returns: pd.DataFrame) -> pd.Series:
    """
    Diversification Ratio (DR) requires constituent asset vols and portfolio weights.

    The OOS returns DataFrame contains only portfolio returns, so DR cannot be
    computed without additional inputs (weights and underlying asset returns).

    Returns NaN for each strategy to avoid a misleading proxy.
    """
    return pd.Series(np.nan, index=returns.columns)


def compute_metrics_table(
    df: pd.DataFrame, rf: float = 0.0, alpha: float = 0.05, input_type: str = "log"
) -> pd.DataFrame:
    """
    Thesis-ready table: rows=strategies, cols=metrics.
    """
    simple = _to_simple_returns(df, input_type=input_type)
    log = df if input_type == "log" else np.log1p(simple)

    out = pd.DataFrame(
        {
            # CAGR is more stable from log returns when input is log.
            "CAGR": _log_sum_cagr(log) if input_type == "log" else compute_cagr(simple),
            "Volatility": compute_volatility(simple),
            "Sharpe": compute_sharpe(simple, rf=rf),
            "Sortino": compute_sortino(simple, rf=rf),
            "MaxDrawdown": compute_max_drawdown(simple),
            "CVaR_5%": compute_cvar(simple, alpha=alpha),
            "DiversificationRatio": compute_diversification_ratio(df),
            "Skewness": compute_skewness(simple),
            "Kurtosis": compute_kurtosis(simple),
        }
    )
    return out


# Backward-compatible function name used by main.py.
def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all performance metrics for each strategy column.

    Returns rows=metrics, cols=strategies (legacy orientation).
    """
    table = compute_metrics_table(df, rf=0.0, alpha=0.05, input_type="log")
    return table.T
