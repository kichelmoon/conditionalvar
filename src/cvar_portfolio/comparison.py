"""
Compare the CVaR pipeline across asset universes.

What theory predicts
--------------------
1. Nested universes, same dates: if A is a subset of B, then W_A (padded with
   zeros) is a subset of W_B. In-sample optima can only improve:
       CVaR*_B <= CVaR*_A   and   Var*_B <= Var*_A.
   Nothing like this holds out of sample.

2. Ties at eta*: at a vertex solution of the CVaR-LP the active constraints
   must number at least n + 1 + T. Count them, with s = #held assets,
   A / B / C = #scenarios with loss above / exactly at / below eta*:
       budget (1) + {w_i = 0} (n - s) + tight scenario rows (A + B) + {z_t = 0} (B + C)
       = 1 + n - s + T + B  >=  n + 1 + T     =>     B >= s.
   So the optimiser puts at least as many scenarios exactly on the kink as it
   holds assets. More assets held means more ties and room for more fractional
   dual atoms.

3. Estimation error: with n free weights and only about (1-alpha)T tail scenarios,
   the optimiser has more room to fit noise as n grows. The out-of-sample over
   in-sample CVaR ratio and the bootstrap instability of the weights measure this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MarketData
from .duality import DualAnalysis
from .metrics import portfolio_metrics
from .optimization import PortfolioResult, equal_weight, optimize_mean_variance, optimize_min_cvar
from .robustness import block_bootstrap_indices, cvar_difference_ci

HELD_TOL = 1e-4


@dataclass
class UniverseRun:
    """All results of the pipeline for one asset universe."""
    name: str
    data: MarketData
    alpha: float
    results: dict[str, PortfolioResult]
    dual: DualAnalysis


def subset_universe(data: MarketData, tickers: list[str]) -> MarketData:
    """
    Restrict a MarketData object to `tickers`, keeping exactly the same dates.
    This makes universes comparable: same T, same train/test split.
    """
    cols = [t for t in tickers if t in data.tickers]
    missing = sorted(set(tickers) - set(cols))
    if missing:
        raise ValueError(f"Tickers not in the larger universe: {missing}")
    idx = [data.tickers.index(t) for t in cols]
    return MarketData(
        tickers=cols,
        prices=data.prices[cols],
        returns=data.returns[cols],
        train_returns=data.train_returns[:, idx],
        test_returns=data.test_returns[:, idx],
        train_dates=data.train_dates,
        test_dates=data.test_dates,
    )


def run_universe(name: str, data: MarketData, alpha: float = 0.95) -> UniverseRun:
    """
    Solve Min-CVaR and Min-Variance on the training data, analyse the dual, and
    add the naive 1/n portfolio as an estimation-free benchmark.
    """
    cvar = optimize_min_cvar(data.train_returns, alpha=alpha)
    mv = optimize_mean_variance(data.train_returns, alpha=alpha)
    ew = equal_weight(data.train_returns, alpha=alpha)
    dual = DualAnalysis(cvar, data.train_dates, data.train_returns, data.tickers)
    return UniverseRun(name, data, alpha,
                       {"Min-CVaR": cvar, "Min-Variance": mv, "1/n": ew}, dual)


def effective_n(w: np.ndarray) -> float:
    """Inverse Herfindahl index 1 / sum w_i^2 (equals n for 1/n, 1 for a single asset)."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


def quality_table(runs: list[UniverseRun]) -> pd.DataFrame:
    """
    One column per (universe, portfolio), rows grouped into
    diversification, in-sample vs out-of-sample risk, performance, and dual structure.
    """
    cols = {}
    for run in runs:
        T = len(run.data.train_dates)
        n = len(run.data.tickers)
        for label, res in run.results.items():
            w = res.weights
            oos = portfolio_metrics(w, run.data.test_returns, run.alpha)
            col = {
                "assets available": n,
                "n / T (train)": n / T,
                f"assets held (w > {HELD_TOL:g})": int((w > HELD_TOL).sum()),
                "effective N (1/sum w^2)": effective_n(w),
                "max weight": float(w.max()),
                "IS CVaR (daily)": res.cvar,
                "OOS CVaR (daily)": oos[f"cvar_{run.alpha:.0%}_daily"],
                "OOS / IS CVaR": oos[f"cvar_{run.alpha:.0%}_daily"] / res.cvar,
                "IS vol (daily)": res.volatility,
                "OOS vol (daily)": oos["ann_vol"] / np.sqrt(252),
                "OOS / IS vol": oos["ann_vol"] / np.sqrt(252) / res.volatility,
                "OOS ann. return": oos["ann_return"],
                "OOS Sharpe": oos["sharpe"],
                "OOS max drawdown": oos["max_drawdown"],
            }
            if label == "Min-CVaR":
                s = run.dual.scenario_concentration()
                col.update({
                    "dual: active scenarios": s["active scenarios (q>0)"],
                    "dual: fractional atoms": s["fractional atoms (0<q<c)"],
                    "dual: ties at eta*": s["scenarios tied at eta*"],
                    "dual: duality gap": s["duality gap"],
                })
            cols[(run.name, label)] = col
    df = pd.DataFrame(cols)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["universe", "portfolio"])
    return df


def check_nesting(small: UniverseRun, large: UniverseRun, tol: float = 1e-7) -> pd.DataFrame:
    """
    Verify the in-sample guarantee CVaR*_large <= CVaR*_small, Var*_large <= Var*_small.
    Requires small.tickers to be a subset of large.tickers on identical dates.
    """
    if not set(small.data.tickers) <= set(large.data.tickers):
        raise ValueError("Universes are not nested.")
    if not small.data.train_dates.equals(large.data.train_dates):
        raise ValueError("Universes have different training dates; use subset_universe().")
    rows = {
        "Min-CVaR: IS CVaR": (small.results["Min-CVaR"].cvar, large.results["Min-CVaR"].cvar),
        "Min-Variance: IS variance": (small.results["Min-Variance"].volatility ** 2,
                                      large.results["Min-Variance"].volatility ** 2),
    }
    df = pd.DataFrame(rows, index=[small.name, large.name]).T
    df["improvement"] = df[small.name] - df[large.name]
    df["guarantee holds"] = df["improvement"] >= -tol
    return df


def bootstrap_instability(
    run: UniverseRun,
    n_boot: int = 30,
    block: int = 20,
    seed: int = 0,
) -> pd.Series:
    """
    Mean L1 distance ||w_boot - w*||_1 (in [0, 2]) between re-optimised and
    full-sample weights, per portfolio. Scale-free, so comparable across n.
    1/n is omitted: it does not depend on the data, so its distance is 0 by construction.
    """
    rng = np.random.default_rng(seed)
    R = run.data.train_returns
    opts = {
        "Min-CVaR": lambda X: optimize_min_cvar(X, alpha=run.alpha).weights,
        "Min-Variance": lambda X: optimize_mean_variance(X, alpha=run.alpha).weights,
    }
    dist = {k: [] for k in opts}
    for _ in range(n_boot):
        X = R[block_bootstrap_indices(len(R), block, rng)]
        for k, opt in opts.items():
            dist[k].append(np.abs(opt(X) - run.results[k].weights).sum())
    return pd.Series({k: float(np.mean(v)) for k, v in dist.items()},
                     name=f"{run.name}: mean ||w_boot - w*||_1")


def oos_cvar_vs_benchmark(run: UniverseRun, portfolio: str, benchmark: str = "1/n",
                          **kwargs) -> dict:
    """Block-bootstrap CI for OOS CVaR(portfolio) - CVaR(benchmark) within one universe, paired by day."""
    R = run.data.test_returns
    return cvar_difference_ci(R @ run.results[portfolio].weights,
                              R @ run.results[benchmark].weights, alpha=run.alpha, **kwargs)


def oos_cvar_difference(run_a: UniverseRun, run_b: UniverseRun, portfolio: str,
                        **kwargs) -> dict:
    """Block-bootstrap CI for OOS CVaR(a) - CVaR(b) of the same portfolio type, paired by day."""
    if not run_a.data.test_dates.equals(run_b.data.test_dates):
        raise ValueError("Universes have different test dates; use subset_universe().")
    ra = run_a.data.test_returns @ run_a.results[portfolio].weights
    rb = run_b.data.test_returns @ run_b.results[portfolio].weights
    return cvar_difference_ci(ra, rb, alpha=run_a.alpha, **kwargs)
