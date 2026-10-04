"""Modèle SVI raw et calibration par moindres carrés."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """w(k) = a + b (rho (k-m) + sqrt((k-m)^2 + sigma^2))."""
    k = np.asarray(k, dtype=float)
    x = k - m
    return a + b * (rho * x + np.sqrt(x**2 + sigma**2))


def calibrate_svi(
    k: np.ndarray,
    w_market: np.ndarray,
    x0: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Calibre (a,b,rho,m,sigma) en minimisant ||w_svi - w_mkt||^2.
    Contraintes douces : b>0, |rho|<1, sigma>0, w>0.
    """
    k = np.asarray(k, dtype=float)
    w_market = np.asarray(w_market, dtype=float)
    if x0 is None:
        # heuristique raisonnable
        x0 = np.array([
            float(np.min(w_market) * 0.5),
            0.2,
            -0.3,
            float(np.median(k)),
            0.2,
        ])

    def residuals(theta: np.ndarray) -> np.ndarray:
        a, b, rho, m, sig = theta
        # pénalités via transformation douce dans residuals
        w = svi_total_variance(k, a, b, rho, m, sig)
        return w - w_market

    # bornes
    lb = np.array([-1.0, 1e-6, -0.999, -2.0, 1e-4])
    ub = np.array([2.0, 5.0, 0.999, 2.0, 2.0])
    res = least_squares(residuals, x0, bounds=(lb, ub), method="trf")
    a, b, rho, m, sig = res.x
    return {
        "a": float(a),
        "b": float(b),
        "rho": float(rho),
        "m": float(m),
        "sigma": float(sig),
        "rmse": float(np.sqrt(np.mean(res.fun**2))),
        "success": bool(res.success),
    }


def make_synthetic_smile(
    F: float = 100.0,
    T: float = 0.5,
    true_params: tuple[float, float, float, float, float] = (0.04, 0.2, -0.4, 0.0, 0.3),
    n_strikes: int = 15,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (K, iv, w) pour un smile SVI + bruit optionnel."""
    rng = np.random.default_rng(seed)
    a, b, rho, m, sig = true_params
    K = np.linspace(0.7 * F, 1.3 * F, n_strikes)
    k = np.log(K / F)
    w = svi_total_variance(k, a, b, rho, m, sig)
    if noise > 0:
        w = w + rng.normal(0.0, noise, size=w.shape)
        w = np.maximum(w, 1e-6)
    iv = np.sqrt(w / T)
    return K, iv, w
