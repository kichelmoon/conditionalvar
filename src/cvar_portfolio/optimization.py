"""
Portfolio optimisers.

  1. CVaR minimisation as a linear program (Rockafellar & Uryasev, 2000)
  2. Mean-variance (Markowitz, 1952)
  3. Equal weight 1/n (naive benchmark, DeMiguel, Garlappi & Uppal, 2009)

Both share the same feasible set W of weights (budget, long-only, box,
optional return target), described by `FeasibleSet`. Keeping W explicit lets
`duality.DualAnalysis` evaluate the Lagrange dual function exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np

from .risk import empirical_cvar, empirical_var, tail_mass

TRADING_DAYS = 252


@dataclass(frozen=True)
class FeasibleSet:
    """W = { w : 1^T w = 1, [w >= 0], [w <= max_weight], [mu^T w >= return_target] }."""
    long_only: bool = True
    max_weight: float = 1.0
    return_target: Optional[float] = None

    def constraints(self, w: cp.Variable, mu: np.ndarray) -> list:
        cons = [cp.sum(w) == 1]
        if self.long_only:
            cons.append(w >= 0)
        if self.max_weight < 1.0:
            cons.append(w <= self.max_weight)
        if self.return_target is not None:
            cons.append(mu @ w >= self.return_target)
        return cons

    @property
    def is_simplex(self) -> bool:
        """True if W is exactly the probability simplex."""
        return self.long_only and self.max_weight >= 1.0 and self.return_target is None


@dataclass
class PortfolioResult:
    """A single portfolio solution. All risk numbers are daily losses (simple returns)."""
    weights: np.ndarray
    alpha: float
    expected_return: float              # daily mean simple return
    cvar: float                         # empirical CVaR_alpha of the in-sample losses
    var: float                          # empirical VaR_alpha (lower quantile)
    volatility: float                   # daily std of portfolio returns
    feasible_set: FeasibleSet = field(default_factory=FeasibleSet)
    eta: Optional[float] = None         # LP threshold eta* (CVaR only), an element of argmin
    dual_scenario_weights: Optional[np.ndarray] = None   # q* (CVaR only)

    @property
    def ann_return(self) -> float:
        """Arithmetic annualisation of the daily mean (simple returns)."""
        return float(self.expected_return * TRADING_DAYS)

    @property
    def ann_vol(self) -> float:
        return float(self.volatility * np.sqrt(TRADING_DAYS))

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio with risk-free rate 0."""
        return self.ann_return / self.ann_vol if self.ann_vol > 0 else np.nan


def _check_status(status: str, solver_name: str) -> None:
    if status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"{solver_name} failed with status '{status}'. "
            "Try relaxing constraints or increasing data."
        )


def _summarise(returns: np.ndarray, w: np.ndarray, alpha: float, feasible: FeasibleSet,
               **extra) -> PortfolioResult:
    port = returns @ w
    losses = -port
    return PortfolioResult(
        weights=w,
        alpha=alpha,
        expected_return=float(port.mean()),
        cvar=empirical_cvar(losses, alpha),
        var=empirical_var(losses, alpha),
        volatility=float(port.std(ddof=1)),
        feasible_set=feasible,
        **extra,
    )


def optimize_min_cvar(
    returns: np.ndarray,
    alpha: float = 0.95,
    return_target: Optional[float] = None,
    long_only: bool = True,
    max_weight: float = 1.0,
) -> PortfolioResult:
    """
    Minimise empirical CVaR_alpha over W via the Rockafellar-Uryasev LP

        min_{w in W, eta, z >= 0}  eta + 1/k * sum_t z_t
        s.t.  z_t >= -r_t^T w - eta      (dual variable q_t >= 0)

    with k = (1 - alpha) T. The duals q* of the scenario constraints form a
    probability measure in the risk envelope {0 <= q <= 1/k, sum q = 1}.
    """
    returns = np.asarray(returns, dtype=float)
    T, n = returns.shape
    if T < 50:
        raise ValueError(f"Too few scenarios: T={T}. Need at least 50.")

    feasible = FeasibleSet(long_only, max_weight, return_target)
    mu = returns.mean(axis=0)
    k = tail_mass(alpha, T)

    w = cp.Variable(n, name="weights")
    eta = cp.Variable(name="eta")
    z = cp.Variable(T, name="excess_losses", nonneg=True)

    scenario_cons = z + eta + returns @ w >= 0
    prob = cp.Problem(
        cp.Minimize(eta + cp.sum(z) / k),
        [scenario_cons, *feasible.constraints(w, mu)],
    )
    prob.solve(solver=cp.CLARABEL)
    _check_status(prob.status, "CVaR-LP")

    w_val = np.asarray(w.value, dtype=float)
    q = scenario_cons.dual_value
    q = None if q is None else np.asarray(q, dtype=float)

    res = _summarise(returns, w_val, alpha, feasible, eta=float(eta.value),
                     dual_scenario_weights=q)
    # The LP optimum and the closed-form empirical CVaR must agree.
    if abs(res.cvar - prob.value) > 1e-6 * max(1.0, abs(prob.value)):
        raise RuntimeError(f"LP value {prob.value} != empirical CVaR {res.cvar}")
    return res


def optimize_mean_variance(
    returns: np.ndarray,
    alpha: float = 0.95,
    return_target: Optional[float] = None,
    long_only: bool = True,
    max_weight: float = 1.0,
    risk_aversion: Optional[float] = None,
) -> PortfolioResult:
    """
    Markowitz mean-variance.

      risk_aversion is None: minimise w^T Sigma w over W (global min variance if no target)
      risk_aversion = lam:   maximise mu^T w - lam/2 * w^T Sigma w over W

    `alpha` is only used to report CVaR / VaR of the resulting portfolio.
    """
    returns = np.asarray(returns, dtype=float)
    feasible = FeasibleSet(long_only, max_weight, return_target)
    mu = returns.mean(axis=0)
    sigma = np.cov(returns.T)

    w = cp.Variable(returns.shape[1], name="weights")
    variance = cp.quad_form(w, cp.psd_wrap(sigma))
    if risk_aversion is None:
        objective = cp.Minimize(variance)
    else:
        objective = cp.Maximize(mu @ w - 0.5 * risk_aversion * variance)

    prob = cp.Problem(objective, feasible.constraints(w, mu))
    prob.solve(solver=cp.CLARABEL)
    _check_status(prob.status, "Mean-Variance")

    return _summarise(returns, np.asarray(w.value, dtype=float), alpha, feasible)


def equal_weight(returns: np.ndarray, alpha: float = 0.95) -> PortfolioResult:
    """
    Naive 1/n portfolio: w_i = 1/n for every asset, no estimation at all.

    It ignores the data except for the number of columns, so it has zero
    estimation error and is the standard benchmark for whether an optimiser
    adds value out of sample. `alpha` is only used to report CVaR / VaR.
    """
    returns = np.asarray(returns, dtype=float)
    n = returns.shape[1]
    return _summarise(returns, np.full(n, 1.0 / n), alpha, FeasibleSet())
