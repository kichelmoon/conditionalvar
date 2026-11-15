"""
Visualize the CVaR portfolio analysis.
Figures are saved to `results/` when `save_path` is supplied.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import data
import optimization
from optimization import *


plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       130,
})

COLORS = {
    "cvar": "#1f4e79",     # dark blue
    "mv":   "#c55a11",     # burnt orange
    "tail": "#c00000",     # red for worst scenarios
    "neutral": "#595959",  # grey
}

def plot_weights(
    cvar_result: PortfolioResult,
    mv_result: PortfolioResult,
    tickers: list[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart: CVaR vs MV portfolio weights."""
    n = len(tickers)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width / 2, cvar_result.weights, width, label="Min-CVaR", color=COLORS["cvar"], alpha=0.85)
    ax.bar(x + width / 2, mv_result.weights, width, label="Min-Variance", color=COLORS["mv"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Portfolio Weight", fontsize=11)
    ax.set_title("Portfolio Weights: CVaR-optimal vs Variance-optimal", fontsize=12)
    ax.legend(fontsize=10)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def _save(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)


market_data = data.fetch_data()
cvar_portfolio = optimization.optimize_min_cvar(market_data.returns)
m_portfolio = optimization.optimize_mean_variance(market_data.returns)
plot_weights(cvar_portfolio, m_portfolio, data.TICKERS, "plot.png")