import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio import (
    backtest_equity,
    efficient_frontier_analytic,
    efficient_frontier_long_only,
    gmv_weights,
    ledoit_wolf_cov,
    max_sharpe_weights,
    optimize_long_only,
    portfolio_stats,
    sample_covariance,
    sharpe_ratio,
    simulate_returns,
)


def test_weights_sum_to_one_long_only():
    _, mu, cov = simulate_returns(seed=1)
    target = float(np.mean(mu))
    w = optimize_long_only(mu, cov, target)
    assert np.isclose(w.sum(), 1.0, atol=1e-5)
    assert np.all(w >= -1e-8)


def test_frontier_shapes():
    _, mu, cov = simulate_returns(seed=2)
    t, v, W = efficient_frontier_analytic(mu, cov, n_points=20)
    assert t.shape == (20,)
    assert v.shape == (20,)
    assert W.shape == (20, len(mu))
    # budget constraint for analytic solution
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-8)


def test_portfolio_stats_positive_vol():
    _, mu, cov = simulate_returns(seed=3)
    w = np.ones(len(mu)) / len(mu)
    ret, vol = portfolio_stats(w, mu, cov)
    assert vol > 0
    assert np.isfinite(ret)


def test_ledoit_wolf_psd_and_shrinks():
    """Résultat du rapport : le shrinkage produit une matrice PSD plus régulière."""
    R, _, _ = simulate_returns(n_assets=8, n_days=120, seed=11)
    S = sample_covariance(R, annualize=False)
    Sigma_lw = ledoit_wolf_cov(R)
    eig_s = np.linalg.eigvalsh(S)
    eig_lw = np.linalg.eigvalsh(Sigma_lw)
    assert np.all(eig_lw >= -1e-10)
    # le conditionnement doit s'améliorer (ratio λ_max/λ_min diminue)
    cond_s = eig_s[-1] / max(eig_s[0], 1e-18)
    cond_lw = eig_lw[-1] / max(eig_lw[0], 1e-18)
    assert cond_lw <= cond_s * 1.01


def test_long_only_frontier_dominates_equal_weight_risk_for_same_return():
    """Pour un rendement proche de l'égal-pondéré, le QP long-only a une vol ≤."""
    _, mu, cov = simulate_returns(seed=5)
    w_eq = np.ones(len(mu)) / len(mu)
    r_eq, v_eq = portfolio_stats(w_eq, mu, cov)
    w = optimize_long_only(mu, cov, target_return=r_eq)
    _, v = portfolio_stats(w, mu, cov)
    assert v <= v_eq + 1e-6


def test_sharpe_and_backtest():
    R, mu, cov = simulate_returns(seed=9)
    w = max_sharpe_weights(mu, cov, rf=0.0, long_only=True)
    assert np.isclose(w.sum(), 1.0, atol=1e-5)
    assert np.all(w >= -1e-8)
    ret, vol = portfolio_stats(w, mu, cov)
    s = sharpe_ratio(ret, vol)
    assert s > 0
    eq = backtest_equity(R, w, rebalance_every=21)
    assert eq.shape == (R.shape[0] + 1,)
    assert eq[0] == 1.0
    assert np.all(eq > 0)


def test_gmv_analytic_minimizes_variance_among_budget():
    _, mu, cov = simulate_returns(seed=4, n_assets=4)
    w_gmv = gmv_weights(cov, long_only=False)
    assert np.isclose(w_gmv.sum(), 1.0)
    _, v_gmv = portfolio_stats(w_gmv, mu, cov)
    # quelques portefeuilles aléatoires sur le simplexe (via normalisation)
    rng = np.random.default_rng(0)
    for _ in range(30):
        w = rng.normal(size=len(mu))
        w = w / w.sum()
        _, v = portfolio_stats(w, mu, cov)
        assert v_gmv <= v + 1e-8


def test_efficient_frontier_long_only_shapes():
    _, mu, cov = simulate_returns(seed=6)
    t, v, W = efficient_frontier_long_only(mu, cov, n_points=12)
    assert len(t) >= 5
    assert W.shape[1] == len(mu)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(W >= -1e-6)
