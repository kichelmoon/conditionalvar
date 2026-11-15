"""
Dual analysis of the CVaR-LP.

Convex-analysis background
--------------------------
With k = (1 - alpha) T and c = 1/k, CVaR is the support function of the
*risk envelope*

    Q = { q in R^T : 0 <= q_t <= c,  sum_t q_t = 1 },
    CVaR_alpha(L) = max_{q in Q} <q, L>,

i.e. its Fenchel conjugate is the indicator of Q. For the portfolio loss
L(w) = -R w this gives the saddle-point problem

    min_{w in W} max_{q in Q} <q, -R w>  =  max_{q in Q} g(q),
    g(q) = min_{w in W} -(R^T q)^T w,

(minimax holds: W, Q compact convex, the pairing is bilinear). g is exactly
the Lagrange dual function of the LP once eta and z are eliminated, and the
LP duals q* of the scenario constraints are a maximiser.

Structure of q*
---------------
Extreme points of Q put mass c on floor(k) scenarios and the remainder
1 - floor(k) c on one more scenario. If several scenarios have loss exactly
eta* at the optimum -- which is *generic* after optimising over w, because the
optimiser pushes scenarios onto the kink -- q* is in general not unique and
can spread fractional mass over all those tied scenarios. So "uniform on the
worst ceil((1-alpha)T) scenarios" is only the special case without ties.

KKT / first-order condition in w (W = simplex)
----------------------------------------------
Writing a_i = E_{Q*}[-r_i] (asset i's expected loss under the worst-case
measure), stationarity + complementary slackness give

    a_i = lambda*  if w_i* > 0,      a_i >= lambda*  if w_i* = 0,

and lambda* = sum_i w_i a_i = E_{Q*}[L(w*)] = CVaR*. All held assets have the
same worst-case expected loss, equal to the optimal CVaR. The products
w_i a_i are the Euler risk contributions (CVaR is positively homogeneous)
and sum exactly to CVaR*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .optimization import PortfolioResult
from .risk import tail_mass

WEIGHT_TOL = 1e-6


@dataclass
class DualAnalysis:
    """
    Analysis of the dual variables q* of a CVaR-LP solution.

    Parameters
    ----------
    result  : PortfolioResult from optimize_min_cvar (dual weights populated)
    dates   : DatetimeIndex aligned with the return scenarios
    returns : (T, n) return matrix used in the optimisation
    tickers : asset names
    tol     : relative tolerance (in units of c) to call q_t "positive" / "full"
    """
    result: PortfolioResult
    dates: pd.DatetimeIndex
    returns: np.ndarray
    tickers: list[str]
    tol: float = 1e-6

    def __post_init__(self) -> None:
        self.returns = np.asarray(self.returns, dtype=float)
        self._T, self._n = self.returns.shape
        self.alpha = self.result.alpha
        self.k = tail_mass(self.alpha, self._T)
        self.c = 1.0 / self.k
        q = self.result.dual_scenario_weights
        if q is None or not np.any(q > 0):
            raise ValueError("PortfolioResult has no dual weights; use optimize_min_cvar.")
        # No silent re-normalisation: deviations from Q are reported, not hidden.
        self.q = np.asarray(q, dtype=float)

    # ------------------------------------------------------------------ basics
    @property
    def losses(self) -> np.ndarray:
        return -(self.returns @ self.result.weights)

    @property
    def scenario_weights(self) -> pd.Series:
        """Dual probability measure q* as a time-indexed Series."""
        return pd.Series(self.q, index=self.dates, name="q_dual")

    def envelope_violation(self) -> dict:
        """How far q* is from the risk envelope Q (should all be ~1e-9 or smaller)."""
        return {
            "|sum q - 1|": float(abs(self.q.sum() - 1.0)),
            "max(q - c, 0)": float(max((self.q - self.c).max(), 0.0)),
            "max(-q, 0)": float(max((-self.q).max(), 0.0)),
        }

    @property
    def tail_scenarios(self) -> pd.DataFrame:
        """Scenarios with q_t > 0, worst loss first, classified as full (q = c) or fractional."""
        eps = self.tol * self.c
        df = pd.DataFrame({
            "date": self.dates,
            "q": self.q,
            "q_over_c": self.q / self.c,
            "portfolio_loss": self.losses,
        })
        tail = df[df["q"] > eps].copy()
        tail["atom"] = np.where(tail["q"] >= self.c - eps, "full", "fractional")
        return tail.sort_values("portfolio_loss", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------- duality checks
    @property
    def expected_loss_under_q(self) -> float:
        """E_{Q*}[L(w*)] = <q*, L(w*)>; equals CVaR* since q* attains the max over Q."""
        return float(self.q @ self.losses)

    @property
    def representation_gap(self) -> float:
        """|CVaR(w*) - E_{Q*}[L(w*)]|: q* is a maximiser in the support-function representation."""
        return abs(self.result.cvar - self.expected_loss_under_q)

    def dual_value(self) -> float:
        """
        Lagrange dual function g(q*) = min_{w in W} <q*, -R w>, solved exactly as a small LP.
        By weak duality g(q*) <= CVaR*; equality certifies optimality of w*.
        """
        a = self.q_expected_losses().to_numpy()
        fs = self.result.feasible_set
        if fs.is_simplex:
            return float(a.min())          # linear function minimised over the simplex
        w = cp.Variable(self._n)
        mu = self.returns.mean(axis=0)
        prob = cp.Problem(cp.Minimize(a @ w), fs.constraints(w, mu))
        prob.solve(solver=cp.CLARABEL)
        return float(prob.value)

    @property
    def duality_gap(self) -> float:
        """Primal minus dual objective: CVaR* - g(q*) (>= 0 by weak duality, ~0 at optimum)."""
        return self.result.cvar - self.dual_value()

    # ---------------------------------------------------------------- KKT
    def q_expected_losses(self) -> pd.Series:
        """a_i = E_{Q*}[-r_i]: each asset's expected loss under the worst-case measure."""
        return pd.Series(-(self.returns.T @ self.q), index=self.tickers, name="E_Q[-r_i]")

    def kkt_table(self) -> pd.DataFrame:
        """
        First-order conditions in w. For W = simplex:
        held assets satisfy a_i = lambda* = CVaR*, the others a_i >= lambda*.
        """
        a = self.q_expected_losses()
        w = pd.Series(self.result.weights, index=self.tickers)
        fs = self.result.feasible_set
        ub = fs.max_weight
        status = np.where(w <= WEIGHT_TOL, "at 0",
                          np.where(w >= ub - WEIGHT_TOL, "at upper bound", "interior"))
        interior = status == "interior"
        lam = self.result.cvar if fs.is_simplex else (
            float(a[interior].median()) if interior.any() else np.nan)
        df = pd.DataFrame({"weight": w, "E_Q[-r_i]": a, "status": status})
        df["E_Q[-r_i] - lambda"] = a - lam
        if fs.return_target is None:
            ok = np.where(status == "interior", np.abs(df["E_Q[-r_i] - lambda"]) < 1e-6,
                 np.where(status == "at 0", df["E_Q[-r_i] - lambda"] > -1e-6,
                          df["E_Q[-r_i] - lambda"] < 1e-6))
            df["kkt_ok"] = ok
        df.attrs["lambda"] = lam
        return df.sort_values("E_Q[-r_i]")

    def risk_contributions(self) -> pd.DataFrame:
        """
        Euler decomposition CVaR* = sum_i w_i E_{Q*}[-r_i].
        (If q* is not unique, this is one element of the subdifferential, not *the* gradient.)
        """
        a = self.q_expected_losses()
        w = pd.Series(self.result.weights, index=self.tickers)
        df = pd.DataFrame({"weight": w, "E_Q[-r_i]": a, "contribution": w * a})
        df["share"] = df["contribution"] / df["contribution"].sum()
        return df.sort_values("contribution", ascending=False)

    # ------------------------------------------------------------ summary
    def scenario_concentration(self) -> dict:
        tail = self.tail_scenarios
        n_full = int((tail["atom"] == "full").sum())
        frac = tail[tail["atom"] == "fractional"]
        n_tied = int(np.sum(np.abs(self.losses - self.result.eta) < 1e-7))
        return {
            "alpha": self.alpha,
            "T": self._T,
            "k = (1-alpha)T": self.k,
            "c = 1/k": self.c,
            "theory: full atoms floor(k)": math.floor(self.k),
            "active scenarios (q>0)": int(len(tail)),
            "full atoms (q=c)": n_full,
            "fractional atoms (0<q<c)": int(len(frac)),
            "mass on fractional atoms": float(frac["q"].sum()),
            "scenarios tied at eta*": n_tied,
            "representation gap": self.representation_gap,
            "duality gap": self.duality_gap,
            **self.envelope_violation(),
        }
