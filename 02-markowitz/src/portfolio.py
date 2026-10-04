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


def sharpe_ratio(ret: float, vol: float, rf: float = 0.0) -> float:
    """Ratio de Sharpe (rf annualisé, même unité que ret/vol)."""
    if vol <= 0:
        return float("nan")
    return float((ret - rf) / vol)


def sample_covariance(returns: np.ndarray, annualize: bool = True) -> np.ndarray:
    """Covariance empirique (optionnellement annualisée, 252 jours)."""
    R = np.asarray(returns, dtype=float)
    cov = np.cov(R, rowvar=False)
    if annualize:
        cov = cov * 252
    return cov


def efficient_frontier_long_only(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Frontière long-only via QP (une cible de rendement par point).
    Retourne (targets, vols, weights[n_points, n_assets]).
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    # Cibles réalisables sous w≥0, 1ᵀw=1 : entre min(μ) et max(μ)
    lo, hi = float(mu.min()), float(mu.max())
    # éviter les extrêmes trop proches où le QP peut être numériquement fragile
    eps = 1e-4 * (hi - lo + 1e-12)
    targets = np.linspace(lo + eps, hi - eps, n_points)
    vols = np.full(n_points, np.nan)
    weights = np.full((n_points, len(mu)), np.nan)
    for i, m in enumerate(targets):
        try:
            w = optimize_long_only(mu, cov, target_return=float(m))
            r, v = portfolio_stats(w, mu, cov)
            targets[i] = r
            vols[i] = v
            weights[i] = w
        except Exception:  # pragma: no cover
            continue
    mask = np.isfinite(vols)
    return targets[mask], vols[mask], weights[mask]


def max_sharpe_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    long_only: bool = True,
) -> np.ndarray:
    """
    Portefeuille de Sharpe maximal.
    - long_only=False : formule tangente (sans contrainte de signe)
    - long_only=True  : balayage sur la frontière QP
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if not long_only:
        excess = mu - rf
        inv = np.linalg.inv(cov)
        w = inv @ excess
        w = w / w.sum()
        return w
    targets, vols, weights = efficient_frontier_long_only(mu, cov, n_points=40)
    sharpes = np.array([sharpe_ratio(t, v, rf) for t, v in zip(targets, vols)])
    i = int(np.nanargmax(sharpes))
    return weights[i]


def backtest_equity(
    returns: np.ndarray,
    weights: np.ndarray,
    rebalance_every: int = 21,
) -> np.ndarray:
    """
    Courbe de richesse d'un portefeuille à pondérations cibles, rebalancé périodiquement.
    returns : (T, n), pondérations : (n,), somme ≈ 1.
    Retourne equity de longueur T+1 (départ à 1).
    """
    R = np.asarray(returns, dtype=float)
    w_target = np.asarray(weights, dtype=float)
    w_target = w_target / w_target.sum()
    T, n = R.shape
    assert n == len(w_target)
    equity = np.ones(T + 1)
    w = w_target.copy()
    for t in range(T):
        # rendement du portefeuille du jour
        r_p = float(w @ R[t])
        equity[t + 1] = equity[t] * (1.0 + r_p)
        # drift des poids
        w = w * (1.0 + R[t])
        w = w / w.sum()
        if (t + 1) % rebalance_every == 0:
            w = w_target.copy()
    return equity


def gmv_weights(cov: np.ndarray, long_only: bool = False) -> np.ndarray:
    """Portefeuille de variance globale minimale (GMV)."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if not long_only:
        inv = np.linalg.inv(cov)
        ones = np.ones(n)
        w = inv @ ones
        return w / w.sum()
    # long-only : QP sans contrainte de rendement (cible très basse)
    mu_dummy = np.zeros(n)
    return optimize_long_only(mu_dummy, cov, target_return=-1e9)
