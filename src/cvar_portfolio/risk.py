"""
Empirical risk measures for a finite scenario set with equal probabilities 1/T.

All functions use the Rockafellar-Uryasev / Acerbi-Tasche definition

    CVaR_alpha(L) = min_eta  eta + 1/((1-alpha) T) * sum_t (L_t - eta)_+ ,

which is the correct (coherent) definition for discrete distributions.
The textbook formula E[L | L >= VaR] coincides with it only when the
distribution has no atom at VaR, which is never guaranteed for empirical data.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["tail_mass", "empirical_var", "empirical_cvar"]


def tail_mass(alpha: float, T: int) -> float:
    """
    k = (1 - alpha) * T, i.e. the number of scenarios the tail "contains".

    Snaps to the nearest integer if k is integral up to floating-point noise:
    (1 - 0.95) * 3020 evaluates to 151.00000000000014, and a naive ceil()
    would turn that into 152.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    k = (1.0 - alpha) * T
    k_round = round(k)
    if abs(k - k_round) < 1e-9 * max(1.0, k):
        return float(k_round)
    return k


def empirical_var(losses: np.ndarray, alpha: float) -> float:
    """
    Lower alpha-quantile of the empirical loss distribution:
    VaR_alpha = min { l : P(L <= l) >= alpha }.

    This is the left end of the argmin interval of the Rockafellar-Uryasev
    function in eta.
    """
    losses = np.asarray(losses, dtype=float)
    T = losses.size
    k = tail_mass(alpha, T)
    j = T - math.floor(k)                 # = ceil(alpha * T), robust version
    return float(np.sort(losses)[j - 1])


def empirical_cvar(losses: np.ndarray, alpha: float) -> float:
    """
    CVaR_alpha of the empirical distribution (closed form of the R-U minimum).

    With k = (1-alpha) T and m = floor(k), losses sorted descending L_(1) >= L_(2) >= ...:

        CVaR = ( L_(1) + ... + L_(m) + (k - m) * L_(m+1) ) / k

    i.e. the full mass 1/k on the m worst scenarios and the remaining
    fraction on the (m+1)-th worst one.
    """
    losses = np.asarray(losses, dtype=float)
    T = losses.size
    k = tail_mass(alpha, T)
    if k < 1e-12:
        return float(losses.max())
    m = math.floor(k)
    desc = np.sort(losses)[::-1]
    head = desc[:m].sum()
    frac = (k - m) * desc[m] if m < T else 0.0
    return float((head + frac) / k)
