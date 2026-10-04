"""Extraction de volatilité implicite (call européen)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def implied_vol_call(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    tol: float = 1e-8,
) -> float:
    """Inverse BS par Brent sur [1e-4, 5]."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price < intrinsic - 1e-12:
        raise ValueError("price below intrinsic")
    # bornes : vol → 0 / ∞
    low, high = 1e-4, 5.0

    def objective(sig: float) -> float:
        return _bs_call(S, K, T, r, sig) - price

    if objective(low) > 0:
        return low
    if objective(high) < 0:
        return high
    return float(brentq(objective, low, high, xtol=tol))
