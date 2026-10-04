"""Pricing call européen par Crank–Nicolson (grille en log-spot)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def price_call_crank_nicolson(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    S_max: float | None = None,
    n_space: int = 200,
    n_time: int = 200,
) -> float:
    """
    Résout l'EDP BS en variable x = log(S) sur [x_min, x_max],
    schéma Crank–Nicolson, condition terminale (S-K)+.
    """
    if S_max is None:
        S_max = max(4.0 * K, 4.0 * S0)
    x_min = np.log(K / 50.0)
    x_max = np.log(S_max)
    x = np.linspace(x_min, x_max, n_space)
    dx = x[1] - x[0]
    dt = T / n_time
    S = np.exp(x)

    # Coefficients constants en x pour BS transformé
    # V_tau = a V_xx + b V_x + c V  avec tau = T-t
    a = 0.5 * sigma**2
    b = r - 0.5 * sigma**2
    c = -r

    alpha = a * dt / (2 * dx**2)
    beta = b * dt / (4 * dx)
    gamma = c * dt / 2

    # Diagonales pour (I - theta L) et (I + theta L), theta=1/2
    # Lower, main, upper for banded solver (l=1, u=1)
    lower = -(alpha - beta) * np.ones(n_space)
    main = (1.0 - gamma + 2 * alpha) * np.ones(n_space)
    upper = -(alpha + beta) * np.ones(n_space)

    # RHS coefficients (explicite)
    lower_r = (alpha - beta) * np.ones(n_space)
    main_r = (1.0 + gamma - 2 * alpha) * np.ones(n_space)
    upper_r = (alpha + beta) * np.ones(n_space)

    V = np.maximum(S - K, 0.0)

    ab = np.zeros((3, n_space))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = main
    ab[2, :-1] = lower[1:]

    for _ in range(n_time):
        rhs = lower_r * np.roll(V, 1) + main_r * V + upper_r * np.roll(V, -1)
        # Bornes : call ~ 0 en S→0, S-K e^{-r tau} asymptotique en S→∞
        # On impose Dirichlet à chaque pas via correction simple
        tau_remaining = None  # non utiliséé; bornes fixes approximatives
        rhs[0] = 0.0
        rhs[-1] = S[-1] - K * np.exp(-r * T)  # borne grossière stable
        ab_step = ab.copy()
        ab_step[1, 0] = 1.0
        ab_step[0, 1] = 0.0
        ab_step[2, 0] = 0.0
        ab_step[1, -1] = 1.0
        ab_step[0, -1] = 0.0
        ab_step[2, -2] = 0.0
        V = solve_banded((1, 1), ab_step, rhs)

    return float(np.interp(np.log(S0), x, V))
