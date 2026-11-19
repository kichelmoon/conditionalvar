"""Universe comparison: nesting guarantee, tie count, table shape."""
import numpy as np
import pandas as pd
import pytest

from cvar_portfolio.comparison import (
    bootstrap_instability, check_nesting, effective_n, oos_cvar_vs_benchmark,
    quality_table, run_universe, subset_universe,
)
from cvar_portfolio.data import TICKERS, TICKERS_100, market_data_from_prices


@pytest.fixture(scope="module")
def universes():
    rng = np.random.default_rng(7)
    T, n = 900, 20
    idx = pd.bdate_range("2015-01-01", periods=T + 1)
    f = 0.008 * rng.standard_t(4, size=(T, 1))
    R = 0.6 * f + 0.01 * rng.standard_t(4, size=(T, n)) + 0.0003
    prices = pd.DataFrame(100 * np.vstack([np.ones(n), np.cumprod(1 + R, axis=0)]),
                          index=idx, columns=[f"S{i:02d}" for i in range(n)])
    big = market_data_from_prices(prices, "2017-12-31")
    small = subset_universe(big, [f"S{i:02d}" for i in range(5)])
    return run_universe("5", small, 0.95), run_universe("20", big, 0.95)


def test_ticker_lists():
    assert len(TICKERS_100) == 100 == len(set(TICKERS_100))
    assert set(TICKERS) <= set(TICKERS_100)


def test_subset_keeps_dates(universes):
    small, big = universes
    assert small.data.train_dates.equals(big.data.train_dates)
    assert np.allclose(small.data.train_returns, big.data.train_returns[:, :5])


def test_nesting_guarantee(universes):
    assert check_nesting(*universes)["guarantee holds"].all()


def test_ties_at_least_held(universes):
    for run in universes:
        held = int((run.results["Min-CVaR"].weights > 1e-6).sum())
        assert run.dual.scenario_concentration()["scenarios tied at eta*"] >= held


def test_effective_n():
    assert effective_n(np.full(8, 1 / 8)) == pytest.approx(8)
    assert effective_n(np.array([1.0, 0, 0])) == pytest.approx(1)


def test_quality_table_and_bootstrap(universes):
    qt = quality_table(list(universes))
    assert qt.shape[1] == 6          # 2 universes x (Min-CVaR, Min-Variance, 1/n)
    assert qt.loc["dual: duality gap"].dropna().abs().max() < 1e-6
    inst = bootstrap_instability(universes[0], n_boot=3)
    assert ((inst >= 0) & (inst <= 2)).all()


def test_equal_weight_benchmark(universes):
    for run in universes:
        n = len(run.data.tickers)
        ew, cvar = run.results["1/n"], run.results["Min-CVaR"]
        assert np.allclose(ew.weights, 1 / n)
        assert effective_n(ew.weights) == pytest.approx(n)
        # 1/n is feasible for the CVaR-LP, so the optimum can only be lower in sample.
        assert cvar.cvar <= ew.cvar + 1e-9
        ci = oos_cvar_vs_benchmark(run, "Min-CVaR")
        assert ci["ci_low"] <= ci["ci_high"]
