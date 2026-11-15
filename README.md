# Conditional Value at Risk using Convex Duality

Minimising the Conditional Value at Risk (CVaR) of a stock portfolio as a linear program, and reading the dual solution as a worst-case probability measure** over historical scenarios.

## Motivation

The first technique usually taught in portfolio optimisation is Markowitz *minimum variance*. Variance is symmetric: a portfolio that outperforms is penalised just as much as one that underperforms. On the plus side, the problem is a convex QP and easy to solve.

The next step is usually *Value at Risk* (VaR), which is used in practice (Basel II). Mathematically it is unpleasant: as a function of the weights it is in general neither convex nor subadditive. Merging positions can increase VaR, and optimising it is a non-convex problem.

*Conditional Value at Risk* (Expected Shortfall) is a coherent risk measure (Artzner et al., 1999). For scenario data its minimisation is a linear program (Rockafellar & Uryasev, 2000). CVaR is the support function of a convex set of probability measures, the *risk envelope*. The LP dual therefore yields the worst-case reweighting of history for free.

## What the project shows

- **CVaR-LP** (Rockafellar–Uryasev) with the correct empirical definition for discrete distributions. Here $\mathbb E[L\mid L\ge \mathrm{VaR}]$ is *not* the right formula, so the code uses the R–U / Acerbi–Tasche definition.
- **Risk envelope and duality.** The dual function $g(q)=\min_{w\in W}\langle q,-Rw\rangle$ is evaluated exactly. Both the duality gap and the representation gap are reported.
- **Structure of $q^*$.** Full atoms ($q_t=c$) are separated from fractional atoms on scenarios tied at $\eta^*$, which explains the non-uniqueness of the dual.
- **KKT certificate.** All held assets have the same expected loss under $Q^*$, and it equals $\mathrm{CVaR}^*$. **Euler risk contributions** add up exactly to $\mathrm{CVaR}^*$.
- **Comparison with min-variance**, in sample and out of sample, with a block-bootstrap confidence interval for the CVaR difference.
- **Robustness:** bootstrap stability of the weights, sensitivity to $\alpha$, and a walk-forward backtest.

The notebook [`analysis.ipynb`](analysis.ipynb) contains the derivations and the results.

## How to use this

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                       # mathematical sanity checks on synthetic data
jupyter lab analysis.ipynb   # first run downloads prices into data_cache/
```

Prices are cached in `data_cache/` (git-ignored), so later runs are reproducible and work offline. Pass `refresh=True` to `fetch_data` to re-download.

## Modelling choices and limitations

- **Simple returns.** The portfolio loss $-R_t^\top w$ is linear in $w$ only for simple returns. Log-returns do not aggregate linearly across assets.
- **Daily risk figures.** VaR and CVaR are reported per day. They are not annualised with $\sqrt{252}$, which is valid only for standard deviations of i.i.d. returns.
- **Survivorship bias.** The universe consists of 14 of *today's* US large caps. That is enough to illustrate the mathematics, but not to support claims about real-world performance.
- **Estimation error.** At $\alpha=0.95$ the CVaR estimate rests on only about 150 tail days. Out of sample, Min-CVaR does not necessarily beat Min-Variance. The notebook discusses and quantifies this.

## References

- R. T. Rockafellar, S. Uryasev (2000). *Optimization of Conditional Value-at-Risk.* Journal of Risk 2(3), 21–41.
- R. T. Rockafellar, S. Uryasev (2002). *Conditional value-at-risk for general loss distributions.* Journal of Banking & Finance 26(7), 1443–1471.
- C. Acerbi, D. Tasche (2002). *On the coherence of expected shortfall.* Journal of Banking & Finance 26(7), 1487–1503.
- P. Artzner, F. Delbaen, J.-M. Eber, D. Heath (1999). *Coherent measures of risk.* Mathematical Finance 9(3), 203–228.
