"""Mathematical sanity checks on synthetic fat-tailed data."""
import numpy as np
import pandas as pd
import pytest

from cvar_portfolio import (
    DualAnalysis, empirical_cvar, empirical_var, optimize_mean_variance,
    optimize_min_cvar, tail_mass,
)
from cvar_portfolio.metrics import portfolio_metrics, rolling_cvar


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(42)
    T, n = 1200, 6
    R = 0.01 * rng.standard_t(4, size=(T, n)) + 0.0003
    R[:, 1] += 0.6 * R[:, 0]          # some correlation
    return R


def ru_objective(losses, alpha):
    """Rockafellar-Uryasev function minimised over eta by brute force on the data points."""
    k = tail_mass(alpha, len(losses))
    return min(eta + np.maximum(losses - eta, 0).sum() / k for eta in losses)


def test_tail_mass_float_noise():
    assert tail_mass(0.95, 3020) == 151.0          # naive ceil gives 152
    assert tail_mass(0.95, 3017) == pytest.approx(150.85)


@pytest.mark.parametrize("alpha", [0.9, 0.95, 0.975, 0.99])
@pytest.mark.parametrize("T", [200, 3017, 3020])
def test_empirical_cvar_matches_ru(alpha, T):
    losses = np.random.default_rng(T).standard_t(3, size=T)
    assert empirical_cvar(losses, alpha) == pytest.approx(ru_objective(losses, alpha), abs=1e-12)


def test_var_is_lower_quantile():
    losses = np.arange(1, 101, dtype=float)       # 1..100
    assert empirical_var(losses, 0.95) == 95.0    # P(L <= 95) = 0.95
    assert empirical_cvar(losses, 0.95) == pytest.approx(98.0)


def test_cvar_dominates_var_and_mean(returns):
    L = -returns[:, 0]
    assert L.mean() <= empirical_var(L, 0.95) <= empirical_cvar(L, 0.95)


def test_lp_value_equals_empirical_cvar(returns):
    res = optimize_min_cvar(returns, 0.95)
    assert res.cvar == pytest.approx(empirical_cvar(-(returns @ res.weights), 0.95))
    assert res.weights.sum() == pytest.approx(1.0)
    assert res.weights.min() > -1e-7


def test_min_cvar_beats_min_variance_in_sample(returns):
    cv = optimize_min_cvar(returns, 0.95)
    mv = optimize_mean_variance(returns, alpha=0.95)
    assert cv.cvar <= mv.cvar + 1e-9
    assert mv.volatility <= cv.volatility + 1e-9
    assert mv.cvar > 0                               # regression: old code averaged the BEST days


@pytest.fixture(scope="module")
def dual(returns):
    res = optimize_min_cvar(returns, 0.95)
    dates = pd.bdate_range("2010-01-01", periods=len(returns))
    return DualAnalysis(res, dates, returns, [f"A{i}" for i in range(returns.shape[1])])


def test_q_in_risk_envelope(dual):
    v = dual.envelope_violation()
    assert v["|sum q - 1|"] < 1e-6
    assert v["max(q - c, 0)"] < 1e-8
    assert v["max(-q, 0)"] < 1e-8


def test_strong_duality(dual):
    assert dual.representation_gap < 1e-7
    assert abs(dual.duality_gap) < 1e-7


def test_kkt_conditions(dual):
    kkt = dual.kkt_table()
    assert kkt["kkt_ok"].all()
    held = kkt[kkt["weight"] > 1e-6]
    assert np.allclose(held["E_Q[-r_i]"], dual.result.cvar, atol=1e-7)


def test_euler_contributions_sum_to_cvar(dual):
    assert dual.risk_contributions()["contribution"].sum() == pytest.approx(dual.result.cvar, abs=1e-8)


def test_dual_structure(dual):
    s = dual.scenario_concentration()
    assert s["full atoms (q=c)"] <= s["theory: full atoms floor(k)"]
    mass = s["full atoms (q=c)"] * s["c = 1/k"] + s["mass on fractional atoms"]
    assert mass == pytest.approx(1.0, abs=1e-6)


def test_box_constraints_duality(returns):
    res = optimize_min_cvar(returns, 0.95, max_weight=0.25)
    dates = pd.bdate_range("2010-01-01", periods=len(returns))
    d = DualAnalysis(res, dates, returns, [f"A{i}" for i in range(returns.shape[1])])
    assert res.weights.max() <= 0.25 + 1e-7
    assert abs(d.duality_gap) < 1e-6
    assert d.kkt_table()["kkt_ok"].all()


def test_metrics_consistency(returns):
    w = np.full(returns.shape[1], 1 / returns.shape[1])
    m = portfolio_metrics(w, returns, 0.95)
    assert m["cvar_95%_daily"] == pytest.approx(empirical_cvar(-(returns @ w), 0.95))
    dates = pd.bdate_range("2010-01-01", periods=len(returns))
    rc = rolling_cvar(w, returns, dates, window=252)
    assert rc.iloc[251] == pytest.approx(empirical_cvar(-(returns[:252] @ w), 0.95))
