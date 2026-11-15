"""
Core portfolio optimizers.

Implements two optimizers:
  1. CVaR minimization via LP  (Rockafellar & Uryasev, 2000) TODO
  2. Mean-Variance (Markowitz, 1952)
"""

from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np

import data


@dataclass
class PortfolioResult:
    """Holds a single portfolio solution."""
    weights: np.ndarray           # (n,) portfolio weights
    expected_return: float        # mean log-return (daily)
    cvar: float                   # CVaR value (daily loss)
    var: float                    # VaR = η* (daily loss)
    volatility: float             # portfolio std dev (daily)
    
    @property
    def ann_return(self) -> float:
        """Annualised expected return (trading days = 252)."""
        return float(np.expm1(self.expected_return * 252))

    @property
    def ann_vol(self) -> float:
        return float(self.volatility * np.sqrt(252))

    @property
    def ann_cvar(self) -> float:
        return float(self.cvar * np.sqrt(252))

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe (risk-free = 0 for simplicity)."""
        return self.ann_return / self.ann_vol if self.ann_vol > 0 else np.nan


# Mean-Variance (Markowitz)
def optimize_mean_variance(
    returns: np.ndarray,
    return_target: Optional[float] = None,
    long_only: bool = True,
    max_weight: float = 1.0,
    risk_aversion: Optional[float] = None,
) -> PortfolioResult:
    """
    Solve the Markowitz Mean-Variance problem.

    Two modes:
      (a) return_target given: minimise variance s.t. E[r^T w] ≥ target
      (b) risk_aversion given: maximise E[r^T w] - (λ/2) Var(r^T w)
      If neither is given, minimise variance (global minimum variance).

    Parameters
    ----------
    returns       : (T, n) array of historical log-returns
    return_target : minimum daily expected return (mode a)
    long_only     : if True enforce w ≥ 0
    max_weight    : maximum weight per asset
    risk_aversion : λ in the utility function (mode b)

    Returns
    -------
    PortfolioResult
    """
    T, n = returns.shape
    mu  = returns.mean(axis=0)
    Sigma = np.cov(returns.T)

    w = cp.Variable(n, name="weights")
    portfolio_variance = cp.quad_form(w, Sigma)

    if risk_aversion is None:
    	objective = cp.Minimize(portfolio_variance)
    else:
        objective = cp.Maximize(mu @ w - 0.5 * risk_aversion * portfolio_variance)


    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight < 1.0:
        constraints.append(w <= max_weight)
    if return_target is not None:
        constraints.append(mu @ w >= return_target)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    _check_status(prob.status, "Mean-Variance")

    w_val = w.value
    portfolio_returns = returns @ w_val

    # Compute CVaR of the resulting portfolio for fair comparison
    alpha = 0.95
    sorted_losses = np.sort(-portfolio_returns)
    cutoff = int(np.ceil((1 - alpha) * T))
    cvar_val = float(sorted_losses[:cutoff].mean())
    var_val  = float(sorted_losses[cutoff - 1])

    return PortfolioResult(
        weights=w_val,
        expected_return=float(mu @ w_val),
        cvar=cvar_val,
        var=var_val,
        volatility=float(np.std(portfolio_returns))
    )

def _check_status(status: str, solver_name: str) -> None:
    if status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"{solver_name} failed with status '{status}'. "
            "Try relaxing constraints or increasing data."
        )
    