"""Markowitz : frontière efficiente, QP long-only, shrinkage."""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    return ret, vol


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """
    Shrinkage simple style Ledoit–Wolf vers μ I.
    Implémentation pédagogique (pas le paper complet).
    """
    X = np.asarray(returns, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / n
    mu = np.trace(S) / p
    target = mu * np.eye(p)
    # intensité de shrinkage heuristique
    d2 = np.linalg.norm(S - target, ord="fro") ** 2
    b2 = 0.0
    for i in range(n):
        row = np.outer(Xc[i], Xc[i]) - S
        b2 += np.linalg.norm(row, ord="fro") ** 2
    b2 /= n**2
    shrinkage = min(1.0, max(0.0, b2 / d2 if d2 > 0 else 1.0))
    return (1 - shrinkage) * S + shrinkage * target


def efficient_frontier_analytic(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Frontière sans contrainte de positivité (solution Lagrange).
    Retourne (targets, vols, weights[n_points, n_assets]).
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    ones = np.ones(n)
    inv = np.linalg.inv(cov)
    A = float(ones @ inv @ ones)
    B = float(ones @ inv @ mu)
    C = float(mu @ inv @ mu)
    Delta = A * C - B**2

    mu_min, mu_max = float(mu.min()), float(mu.max())
    # élargir un peu la plage autour des actifs
    span = mu_max - mu_min
    targets = np.linspace(mu_min - 0.2 * span, mu_max + 0.2 * span, n_points)
    vols = np.zeros(n_points)
    weights = np.zeros((n_points, n))
    for i, m in enumerate(targets):
        # w = g + h * m
        g = (C * inv @ ones - B * inv @ mu) / Delta
        h = (A * inv @ mu - B * inv @ ones) / Delta
        w = g + h * m
        weights[i] = w
        vols[i] = np.sqrt(max(w @ cov @ w, 0.0))
    return targets, vols, weights


def optimize_long_only(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
) -> np.ndarray:
    """Portefeuille long-only de variance minimale pour un rendement cible."""
    if cp is None:
        raise ImportError("cvxpy is required for optimize_long_only")
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [
        w >= 0,
        cp.sum(w) == 1,
        mu @ w >= target_return,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)
    if w.value is None:
        # fallback SCS
        prob.solve(solver=cp.SCS)
    if w.value is None:
        raise RuntimeError("QP failed to find a solution")
    weights = np.asarray(w.value, dtype=float)
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    return weights


def simulate_returns(
    n_assets: int = 5,
    n_days: int = 756,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Génère des rendements corrélés synthétiques. Retourne (R, mu, cov)."""
    rng = np.random.default_rng(seed)
    # Facteur commun + bruit idiosyncratique
    factor = rng.normal(0.0004, 0.01, size=n_days)
    betas = rng.uniform(0.5, 1.5, size=n_assets)
    idio = rng.normal(0.0, 0.012, size=(n_days, n_assets))
    R = factor[:, None] * betas[None, :] + idio
    mu = R.mean(axis=0) * 252
    cov = np.cov(R, rowvar=False) * 252
    return R, mu, cov
