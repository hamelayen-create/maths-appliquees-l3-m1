"""Monte Carlo Black–Scholes pour call européen."""

from __future__ import annotations

import numpy as np


def price_call_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Retourne (estimateur, erreur_std_approx).
    Si antithetic=True, utilise Z et -Z (n_paths doit être pair idéalement).
    """
    rng = np.random.default_rng(seed)
    n = n_paths if not antithetic else n_paths // 2
    Z = rng.standard_normal(n)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)

    def payoff(z: np.ndarray) -> np.ndarray:
        ST = S0 * np.exp(drift + diffusion * z)
        return np.exp(-r * T) * np.maximum(ST - K, 0.0)

    pay = payoff(Z)
    if antithetic:
        pay = 0.5 * (pay + payoff(-Z))

    estimate = float(np.mean(pay))
    stderr = float(np.std(pay, ddof=1) / np.sqrt(len(pay)))
    return estimate, stderr
