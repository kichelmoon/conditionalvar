"""
Matplotlib figures.

Note on dates: matplotlib stores datetimes as floats (days since 1970-01-01).
Never put a date axis and a plain numeric axis (e.g. scenario ranks) into one
`sharex=True` group -- the numbers 1..T would be drawn as days in 1970.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .duality import DualAnalysis
from .metrics import rolling_cvar, wealth_path
from .optimization import PortfolioResult

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
})

COLORS = {
    "cvar": "#1f4e79",     # dark blue
    "mv": "#c55a11",       # burnt orange
    "tail": "#c00000",     # red for tail scenarios
    "frac": "#e69f00",     # amber for fractional atoms
    "neutral": "#595959",  # grey
}
PORTFOLIO_STYLE = {
    "Min-CVaR": (COLORS["cvar"], "-"),
    "Min-Variance": (COLORS["mv"], "--"),
}

CRISES = [
    ("2008-09-01", "2009-03-31", "GFC"),
    ("2011-08-01", "2011-10-31", "Euro debt"),
    ("2015-08-17", "2015-09-30", "China"),
    ("2018-02-01", "2018-02-28", "Volmageddon"),
    ("2018-10-01", "2018-12-31", "Q4 2018"),
    ("2020-02-20", "2020-04-30", "COVID"),
    ("2022-01-01", "2022-10-31", "Rate hikes"),
    ("2025-04-01", "2025-04-30", "Tariffs"),
]


def _style(label: str, i: int, styles: Optional[dict] = None) -> tuple[str, str]:
    if styles and label in styles:
        return styles[label]
    return PORTFOLIO_STYLE.get(label, (f"C{i}", "-"))


def universe_styles(labels: list[str]) -> dict[str, tuple[str, str]]:
    """
    Styles for labels like "Min-CVaR (14)": colour by portfolio type,
    line style by universe (first universe solid, second dotted, ...).
    """
    universes = []
    for lab in labels:
        u = lab.rsplit("(", 1)[-1].rstrip(")") if "(" in lab else ""
        if u not in universes:
            universes.append(u)
    dashes = ["-", ":", "-.", "--"]
    out = {}
    for i, lab in enumerate(labels):
        base = lab.split(" (")[0]
        color = PORTFOLIO_STYLE.get(base, (f"C{i}", "-"))[0]
        u = lab.rsplit("(", 1)[-1].rstrip(")") if "(" in lab else ""
        out[lab] = (color, dashes[universes.index(u) % len(dashes)])
    return out


def _shade_crises(axes, dates: pd.DatetimeIndex, label_axis=None) -> None:
    """Shade only crises that intersect the plotted date range."""
    lo, hi = dates.min(), dates.max()
    for start, end, label in CRISES:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if e < lo or s > hi:
            continue
        s, e = max(s, lo), min(e, hi)
        for ax in axes:
            ax.axvspan(s, e, alpha=0.08, color="grey", zorder=0, lw=0)
        if label_axis is not None:
            label_axis.text(s + (e - s) / 2, 1.0, label, transform=label_axis.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=7, color="grey", style="italic")


def _save(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)


# ---------------------------------------------------------------- duality
def plot_dual_weights(dual: DualAnalysis, save_path: Optional[str] = None) -> plt.Figure:
    """Dual measure q* over time and the daily losses; both on a shared *date* axis."""
    q = dual.scenario_weights
    losses = pd.Series(dual.losses, index=dual.dates)
    eps = dual.tol * dual.c
    full = q >= dual.c - eps
    frac = (q > eps) & ~full

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

    ax = axes[0]
    ax.vlines(q.index[full], 0, q[full], color=COLORS["tail"], lw=0.9, label="full atom $q_t^*=c$")
    ax.vlines(q.index[frac], 0, q[frac], color=COLORS["frac"], lw=1.6,
              label="fractional atom $0<q_t^*<c$")
    ax.axhline(dual.c, color=COLORS["neutral"], lw=1.0, ls=":", label=f"$c = 1/((1-\\alpha)T)$ = {dual.c:.4f}")
    ax.set_ylim(0, dual.c * 1.15)
    ax.set_ylabel("Dual weight $q_t^*$")
    ax.set_title("Worst-case probability measure $Q^*$ (dual of the CVaR-LP)", fontsize=11, pad=14)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    colors = np.where(full, COLORS["tail"], np.where(frac, COLORS["frac"], COLORS["neutral"]))
    ax.bar(losses.index, losses.values, color=colors, width=1.5, linewidth=0, alpha=0.8)
    ax.axhline(dual.result.var, color="black", lw=1.2, ls="--",
               label=f"VaR {dual.alpha:.0%} = {dual.result.var:.4f}")
    ax.axhline(dual.result.cvar, color=COLORS["cvar"], lw=1.2,
               label=f"CVaR {dual.alpha:.0%} = {dual.result.cvar:.4f}")
    ax.set_ylabel("Daily loss $-r_t^\\top w^*$")
    ax.set_title("Daily losses (red: $q_t^*=c$, amber: fractional)", fontsize=11, pad=14)
    ax.legend(fontsize=8, loc="upper left")

    _shade_crises(axes, dual.dates, label_axis=axes[0])
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dual_mass(dual: DualAnalysis, save_path: Optional[str] = None) -> plt.Figure:
    """Sorted dual weights and cumulative mass vs scenario rank (own, numeric x-axis)."""
    q_sorted = np.sort(dual.q)[::-1]
    n_show = int(min(len(q_sorted), max(2 * dual.k, dual.k + 40)))
    ranks = np.arange(1, n_show + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    ax.step(ranks, q_sorted[:n_show], where="mid", color=COLORS["cvar"], lw=1.8)
    ax.axhline(dual.c, color=COLORS["neutral"], ls=":", lw=1, label="$c$")
    ax.axvline(dual.k, color=COLORS["tail"], ls="--", lw=1.2, label=f"$k=(1-\\alpha)T$ = {dual.k:g}")
    ax.set_xlabel("Scenario rank (sorted by $q_t^*$)")
    ax.set_ylabel("$q_t^*$")
    ax.set_title("Sorted dual weights", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ranks, np.cumsum(q_sorted)[:n_show], color=COLORS["cvar"], lw=2)
    ax.plot([0, dual.k], [0, 1], color=COLORS["neutral"], ls=":", lw=1,
            label="extreme point of $\\mathcal{Q}$ (no ties)")
    ax.axvline(dual.k, color=COLORS["tail"], ls="--", lw=1.2, label=f"$k$ = {dual.k:g}")
    ax.set_xlabel("Scenario rank (sorted by $q_t^*$)")
    ax.set_ylabel("Cumulative dual mass")
    ax.set_title("Cumulative mass (linear in rank up to the tied scenarios)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_kkt(dual: DualAnalysis, save_path: Optional[str] = None) -> plt.Figure:
    """Worst-case expected loss per asset; held assets must sit exactly on lambda* = CVaR*."""
    df = dual.kkt_table()
    lam = df.attrs["lambda"]
    held = df["weight"] > 1e-6

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(df))))
    ax.barh(df.index, df["E_Q[-r_i]"], color=np.where(held, COLORS["cvar"], COLORS["neutral"]),
            alpha=0.85)
    ax.axvline(lam, color=COLORS["tail"], ls="--", lw=1.5, label=f"$\\lambda^*$ = CVaR* = {lam:.5f}")
    for y, (w, a) in enumerate(zip(df["weight"], df["E_Q[-r_i]"])):
        if w > 1e-6:
            ax.text(a, y, f"  w = {w:.1%}", va="center", fontsize=8)
    lo = df["E_Q[-r_i]"].min()
    ax.set_xlim(lo * 0.9, df["E_Q[-r_i]"].max() * 1.05)
    ax.set_xlabel("$E_{Q^*}[-r_i]$ (expected loss of asset $i$ under the worst-case measure)")
    ax.set_title("KKT check: held assets (blue) all have $E_{Q^*}[-r_i]=\\lambda^*$, the others $\\geq \\lambda^*$",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_risk_contributions(dual: DualAnalysis, save_path: Optional[str] = None) -> plt.Figure:
    """Euler decomposition CVaR* = sum_i w_i E_{Q*}[-r_i] (held assets only)."""
    df = dual.risk_contributions()
    df = df[df["weight"] > 1e-6].sort_values("contribution")
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(df))))
    ax.barh(df.index, df["contribution"], color=COLORS["tail"], alpha=0.85)
    for y, (c, s) in enumerate(zip(df["contribution"], df["share"])):
        ax.text(c, y, f"  {s:.1%}", va="center", fontsize=8)
    ax.set_xlabel("Contribution to daily CVaR")
    ax.set_title(f"Euler risk contributions (sum = CVaR* = {df['contribution'].sum():.5f})", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ------------------------------------------------------------ portfolios
def plot_weights(results: dict[str, PortfolioResult], tickers: list[str],
                 save_path: Optional[str] = None) -> plt.Figure:
    """Grouped bar chart of portfolio weights."""
    n, m = len(tickers), len(results)
    x = np.arange(n)
    width = 0.8 / m
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (label, res) in enumerate(results.items()):
        color, _ = _style(label, i)
        ax.bar(x + (i - (m - 1) / 2) * width, np.clip(res.weights, 0, None), width,
               label=label, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Optimal portfolio weights (training period)", fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_oos_performance(results: dict[str, PortfolioResult], test_returns: np.ndarray,
                         test_dates: pd.DatetimeIndex, alpha: float = 0.95, window: int = 126,
                         save_path: Optional[str] = None) -> plt.Figure:
    """Out-of-sample wealth and rolling empirical CVaR."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for i, (label, res) in enumerate(results.items()):
        color, ls = _style(label, i)
        wp = wealth_path(res.weights, test_returns, test_dates)
        axes[0].plot(wp.index, wp.values, lw=2, color=color, ls=ls, label=label)
        rc = rolling_cvar(res.weights, test_returns, test_dates, window=window, alpha=alpha)
        axes[1].plot(rc.index, rc.values, lw=2, color=color, ls=ls, label=label)

    axes[0].axhline(1.0, color="grey", lw=0.8, ls=":")
    axes[0].set_ylabel("Value (start = 1)")
    axes[0].set_title("Out-of-sample performance (fixed in-sample weights)", fontsize=12, pad=14)
    axes[0].legend(fontsize=10)
    axes[1].set_ylabel(f"Rolling CVaR {alpha:.0%} ({window}d)")
    axes[1].set_title("Rolling tail risk", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _shade_crises(axes, test_dates, label_axis=axes[0])
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ------------------------------------------------------------ robustness
def plot_bootstrap_weights(boot: dict[str, pd.DataFrame], save_path: Optional[str] = None) -> plt.Figure:
    """Distribution of re-optimised weights across bootstrap resamples."""
    labels = list(boot)
    tickers = list(next(iter(boot.values())).columns)
    n, m = len(tickers), len(labels)
    width = 0.8 / m
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for i, label in enumerate(labels):
        color, _ = _style(label, i)
        pos = np.arange(n) + (i - (m - 1) / 2) * width
        bp = ax.boxplot([boot[label][t].clip(lower=0) for t in tickers], positions=pos,
                        widths=width * 0.85, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.5, edgecolor=color)
        for key in ("whiskers", "caps", "medians"):
            for line in bp[key]:
                line.set_color(color)
        ax.plot([], [], color=color, lw=6, alpha=0.5, label=label)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(tickers, rotation=35, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Weight")
    ax.set_title("Estimation error: optimal weights across block-bootstrap resamples", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_alpha_sensitivity(sens: pd.DataFrame, tickers: list[str],
                           save_path: Optional[str] = None) -> plt.Figure:
    """Stacked area of min-CVaR weights as alpha varies."""
    w = sens[tickers].clip(lower=0)
    w = w.loc[:, w.max() > 1e-4]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[: w.shape[1]]
    axes[0].stackplot(w.index, w.T.values, labels=w.columns, colors=colors, alpha=0.9)
    axes[0].set_xlabel("$\\alpha$")
    axes[0].set_ylabel("Weight")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Min-CVaR weights vs confidence level", fontsize=11)
    axes[0].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    axes[1].plot(sens.index, sens["cvar"], marker="o", color=COLORS["cvar"], label="CVaR$_\\alpha$")
    axes[1].plot(sens.index, sens["var"], marker="s", color="black", ls="--", label="VaR$_\\alpha$")
    axes[1].set_xlabel("$\\alpha$")
    axes[1].set_ylabel("Daily loss")
    axes[1].set_title("Optimal risk vs confidence level", fontsize=11)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_backtest(port_returns: pd.DataFrame, alpha: float = 0.95, window: int = 252,
                  lookback: int = 756, rebalance: int = 63,
                  styles: Optional[dict] = None,
                  save_path: Optional[str] = None) -> plt.Figure:
    """Walk-forward backtest: wealth and rolling CVaR per strategy."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ones = np.ones(1)
    for i, label in enumerate(port_returns.columns):
        color, ls = _style(label, i, styles)
        r = port_returns[label]
        axes[0].plot(r.index, (1 + r).cumprod(), color=color, ls=ls, lw=2, label=label)
        rc = rolling_cvar(ones, r.to_numpy()[:, None], r.index, window=window, alpha=alpha)
        axes[1].plot(rc.index, rc.values, color=color, ls=ls, lw=2, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Value (log scale)")
    axes[0].set_title(f"Walk-forward backtest (re-optimised every {rebalance} days on the trailing {lookback} days)",
                      fontsize=12, pad=14)
    axes[0].legend(fontsize=10)
    axes[1].set_ylabel(f"Rolling CVaR {alpha:.0%} ({window}d)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
    axes[1].legend(fontsize=10)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _shade_crises(axes, port_returns.index, label_axis=axes[0])
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ------------------------------------------------------------ universes
def plot_universe_comparison(runs, window: int = 126,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Four panels comparing universes (runs: list of comparison.UniverseRun):
      (a) in-sample vs out-of-sample CVaR,
      (b) concentration curves: cumulative weight of the k largest positions,
      (c) out-of-sample wealth,
      (d) out-of-sample rolling CVaR.
    """
    from .risk import empirical_cvar

    labels, entries = [], []
    for run in runs:
        for name, res in run.results.items():
            lab = f"{name} ({run.name})"
            labels.append(lab)
            entries.append((lab, run, res))
    styles = universe_styles(labels)
    alpha = runs[0].alpha
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    x = np.arange(len(entries))
    is_c = [res.cvar for _, _, res in entries]
    oos_c = [empirical_cvar(-(run.data.test_returns @ res.weights), alpha) for _, run, res in entries]
    cols = [styles[lab][0] for lab, _, _ in entries]
    ax.bar(x - 0.2, is_c, 0.38, color=cols, alpha=0.45, label="in-sample")
    ax.bar(x + 0.2, oos_c, 0.38, color=cols, alpha=0.95, label="out-of-sample")
    for xi, a, b in zip(x, is_c, oos_c):
        ax.text(xi + 0.2, b, f"{b / a:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace(" (", "\n(") for lab in labels], fontsize=8)
    ax.set_ylabel(f"CVaR {alpha:.0%} (daily)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
    ax.set_title("In-sample vs out-of-sample CVaR (label: OOS / IS)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[0, 1]
    for lab, run, res in entries:
        w = np.sort(np.clip(res.weights, 0, None))[::-1]
        cum = np.cumsum(w) / w.sum()
        k = int(np.searchsorted(cum, 0.999)) + 2
        color, ls = styles[lab]
        ax.step(np.arange(1, min(k, len(w)) + 1), cum[:k], where="post", color=color, ls=ls, lw=2, label=lab)
    ax.set_xlabel("Number of largest positions")
    ax.set_ylabel("Cumulative weight")
    ax.set_title("Concentration curves (training weights)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 0]
    ax_r = axes[1, 1]
    for lab, run, res in entries:
        color, ls = styles[lab]
        wp = wealth_path(res.weights, run.data.test_returns, run.data.test_dates)
        ax.plot(wp.index, wp.values, color=color, ls=ls, lw=1.8, label=lab)
        rc = rolling_cvar(res.weights, run.data.test_returns, run.data.test_dates, window=window, alpha=alpha)
        ax_r.plot(rc.index, rc.values, color=color, ls=ls, lw=1.8, label=lab)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("Value (start = 1)")
    ax.set_title("Out-of-sample wealth (fixed training weights)", fontsize=11)
    ax.legend(fontsize=8)
    ax_r.set_ylabel(f"Rolling CVaR {alpha:.0%} ({window}d)")
    ax_r.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))
    ax_r.set_title("Out-of-sample rolling tail risk", fontsize=11)
    ax_r.legend(fontsize=8)
    for a in (ax, ax_r):
        a.xaxis.set_major_locator(mdates.YearLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    _save(fig, save_path)
    return fig
