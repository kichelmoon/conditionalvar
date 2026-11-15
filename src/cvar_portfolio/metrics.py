"""
Performance and risk metrics on simple daily returns.

Risk numbers (VaR, CVaR) are reported as *daily* losses. They are deliberately
not annualised with sqrt(252): square-root-of-time scaling is only justified
for the standard deviation of i.i.d. returns, not for tail quantiles of
fat-tailed, dependent data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .optimization import TRADING_DAYS, PortfolioResult
from .risk import empirical_cvar, empirical_var


def portfolio_metrics(
    weights: np.ndarray,
    returns: np.ndarray,
    alpha: float = 0.95,
) -> dict:
    """
    Performance and risk metrics of a constant-weight (daily rebalanced) portfolio.
    """
    r = np.asarray(returns) @ weights
    T = len(r)

    mu_ann = float(r.mean() * TRADING_DAYS)
    vol_ann = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    downside_dev = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2) * TRADING_DAYS))

    wealth = np.cumprod(1.0 + r)
    cagr = float(wealth[-1] ** (TRADING_DAYS / T) - 1.0)
    running_max = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    max_dd = float((wealth / running_max - 1.0).min())

    losses = -r
    var = empirical_var(losses, alpha)
    cvar = empirical_cvar(losses, alpha)

    return {
        "ann_return": mu_ann,
        "cagr": cagr,
        "ann_vol": vol_ann,
        "sharpe": mu_ann / vol_ann if vol_ann > 0 else np.nan,
        "sortino": mu_ann / downside_dev if downside_dev > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        f"var_{alpha:.0%}_daily": var,
        f"cvar_{alpha:.0%}_daily": cvar,
        "cvar_to_var": cvar / var if var > 0 else np.nan,
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "n_obs": T,
    }


def out_of_sample_evaluation(
    results: dict[str, PortfolioResult],
    test_returns: np.ndarray,
    alpha: float = 0.95,
) -> pd.DataFrame:
    """One row per portfolio, metrics evaluated on the held-out returns."""
    rows = {label: portfolio_metrics(res.weights, test_returns, alpha)
            for label, res in results.items()}
    df = pd.DataFrame(rows).T
    df.index.name = "Portfolio"
    return df


def rolling_cvar(
    weights: np.ndarray,
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    window: int = 252,
    alpha: float = 0.95,
) -> pd.Series:
    """Rolling empirical CVaR over the trailing `window` days (inclusive of day t)."""
    losses = pd.Series(-(np.asarray(returns) @ weights), index=dates)
    out = losses.rolling(window).apply(lambda x: empirical_cvar(x, alpha), raw=True)
    out.name = f"Rolling CVaR {alpha:.0%} ({window}d)"
    return out


def wealth_path(
    weights: np.ndarray,
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    initial_value: float = 1.0,
) -> pd.Series:
    """Cumulative value of a constant-weight (daily rebalanced) portfolio."""
    r = np.asarray(returns) @ weights
    return pd.Series(initial_value * np.cumprod(1.0 + r), index=dates, name="Portfolio Value")
