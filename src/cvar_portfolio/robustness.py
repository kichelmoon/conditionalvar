"""
Robustness diagnostics.

Why this matters: the CVaR estimate at alpha = 95% rests on only (1-alpha)T
tail observations (about 150 days in 12 years), while the covariance matrix
uses all T observations. Min-CVaR weights are therefore much noisier than
min-variance weights, and that estimation error can more than offset the
"better" objective out of sample. These tools make that visible.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .optimization import PortfolioResult
from .risk import empirical_cvar

Optimizer = Callable[[np.ndarray], PortfolioResult]


def block_bootstrap_indices(T: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap: keeps short-range dependence / volatility clusters."""
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % T
    return idx.ravel()[:T]


def bootstrap_weights(
    returns: np.ndarray,
    optimizer: Optimizer,
    tickers: list[str],
    n_boot: int = 100,
    block: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Re-optimise on block-bootstrap resamples; one row of weights per resample."""
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns)
    rows = [optimizer(returns[block_bootstrap_indices(len(returns), block, rng)]).weights
            for _ in range(n_boot)]
    return pd.DataFrame(rows, columns=tickers)


def alpha_sensitivity(
    returns: np.ndarray,
    alphas: list[float],
    tickers: list[str],
    optimizer: Callable[[np.ndarray, float], PortfolioResult],
) -> pd.DataFrame:
    """Optimal weights and CVaR as a function of the confidence level alpha."""
    rows = {}
    for a in alphas:
        res = optimizer(returns, a)
        rows[a] = {**dict(zip(tickers, res.weights)), "cvar": res.cvar, "var": res.var}
    df = pd.DataFrame(rows).T
    df.index.name = "alpha"
    return df


def rolling_backtest(
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    optimizers: dict[str, Optimizer],
    lookback: int = 756,
    rebalance: int = 63,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Walk-forward backtest: every `rebalance` days re-optimise on the trailing
    `lookback` days and hold those weights (rebalanced daily to the targets)
    until the next rebalancing date. Uses only past data at every step.

    Returns (daily portfolio returns per strategy, weight history per strategy).
    """
    returns = np.asarray(returns)
    T = len(returns)
    rebal_points = list(range(lookback, T, rebalance))
    port = {name: np.full(T, np.nan) for name in optimizers}
    hist = {name: [] for name in optimizers}
    for start in rebal_points:
        window = returns[start - lookback:start]
        stop = min(start + rebalance, T)
        for name, opt in optimizers.items():
            w = opt(window).weights
            port[name][start:stop] = returns[start:stop] @ w
            hist[name].append(pd.Series(w, name=dates[start]))
    port_df = pd.DataFrame(port, index=dates).iloc[lookback:]
    hist_df = {name: pd.DataFrame(h) for name, h in hist.items()}
    return port_df, hist_df


def cvar_difference_ci(
    r_a: np.ndarray,
    r_b: np.ndarray,
    alpha: float = 0.95,
    n_boot: int = 1000,
    block: int = 20,
    seed: int = 0,
) -> dict:
    """
    Block-bootstrap confidence interval for CVaR(A) - CVaR(B) on the *same* days
    (paired resampling). Tells whether an out-of-sample CVaR difference is noise.
    """
    rng = np.random.default_rng(seed)
    r_a, r_b = np.asarray(r_a), np.asarray(r_b)
    point = empirical_cvar(-r_a, alpha) - empirical_cvar(-r_b, alpha)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_bootstrap_indices(len(r_a), block, rng)
        diffs[i] = empirical_cvar(-r_a[idx], alpha) - empirical_cvar(-r_b[idx], alpha)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p_diff_le_0": float(np.mean(diffs <= 0))}
