"""Extraction de volatilité implicite (call européen)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Prix Black–Scholes d'un call européen (taux constant, dividendes nuls)."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


# Alias interne conservé pour compatibilité
_bs_call = bs_call


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega Black–Scholes (dérivée du prix call par rapport à sigma)."""
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * norm.pdf(d1) * np.sqrt(T))


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
    low, high = 1e-4, 5.0

    def objective(sig: float) -> float:
        return bs_call(S, K, T, r, sig) - price

    if objective(low) > 0:
        return low
    if objective(high) < 0:
        return high
    return float(brentq(objective, low, high, xtol=tol))


def implied_vol_newton(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma0: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """
    Inverse BS par Newton–Raphson (utilise la vega).
    Moins robuste que Brent loin de la monnaie ; utile pour la comparaison pédagogique.
    """
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price < intrinsic - 1e-12:
        raise ValueError("price below intrinsic")
    sigma = float(max(sigma0, 1e-4))
    for _ in range(max_iter):
        diff = bs_call(S, K, T, r, sigma) - price
        if abs(diff) < tol:
            return sigma
        v = bs_vega(S, K, T, r, sigma)
        if v < 1e-14:
            break
        sigma = float(np.clip(sigma - diff / v, 1e-4, 5.0))
    # repli robuste
    return implied_vol_call(price, S, K, T, r, tol=tol)