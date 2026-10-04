"""Formules fermées Black–Scholes pour options européennes."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, sigma must be positive")
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(d1), float(d2)


def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = d1_d2(S, K, T, r, sigma)
    return float(norm.cdf(d1))
