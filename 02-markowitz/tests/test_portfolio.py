import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio import (
    efficient_frontier_analytic,
    optimize_long_only,
    portfolio_stats,
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
