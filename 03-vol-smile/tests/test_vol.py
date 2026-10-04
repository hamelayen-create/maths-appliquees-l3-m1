import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from implied_vol import bs_call, implied_vol_call, implied_vol_newton
from svi import (
    butterfly_arbitrage_free,
    build_svi_surface,
    calibrate_svi,
    make_synthetic_smile,
    svi_total_variance,
)


def _bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def test_implied_vol_roundtrip():
    S, K, T, r, sigma = 100.0, 105.0, 0.75, 0.02, 0.22
    price = _bs_call(S, K, T, r, sigma)
    iv = implied_vol_call(price, S, K, T, r)
    assert abs(iv - sigma) < 1e-6


def test_svi_calibration_recovers_params():
    F, T = 100.0, 1.0
    true = (0.04, 0.2, -0.4, 0.0, 0.3)
    K, _, w = make_synthetic_smile(F=F, T=T, true_params=true, noise=0.0)
    k = np.log(K / F)
    fit = calibrate_svi(k, w, x0=np.array(true))
    w_fit = svi_total_variance(k, fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"])
    assert fit["rmse"] < 1e-4
    assert np.max(np.abs(w_fit - w)) < 1e-3


def test_report_demo_rmse_bound():
    """Résultat numérique du rapport / démo : RMSE de total variance très petite."""
    S, r, T = 100.0, 0.01, 0.5
    F = S * np.exp(r * T)
    true = (0.03, 0.25, -0.35, 0.05, 0.25)
    K, _, w_true = make_synthetic_smile(F=F, T=T, true_params=true, noise=1e-4, seed=0)
    k = np.log(K / F)
    fit = calibrate_svi(k, w_true)
    w_fit = svi_total_variance(k, fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"])
    rmse = float(np.sqrt(np.mean((w_fit - w_true) ** 2)))
    assert fit["success"]
    assert rmse < 5e-4


def test_newton_matches_brent_near_atm():
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.01, 0.2
    price = bs_call(S, K, T, r, sigma)
    iv_b = implied_vol_call(price, S, K, T, r)
    iv_n = implied_vol_newton(price, S, K, T, r, sigma0=0.25)
    assert abs(iv_b - sigma) < 1e-7
    assert abs(iv_n - sigma) < 1e-7


def test_butterfly_proxy_on_healthy_svi():
    k = np.linspace(-0.5, 0.5, 101)
    assert butterfly_arbitrage_free(k, 0.03, 0.25, -0.35, 0.05, 0.25)


def test_surface_calendar_projection():
    maturities = np.array([0.5, 1.0])
    params = [(0.04, 0.3, -0.4, 0.0, 0.2), (0.01, 0.1, -0.2, 0.0, 0.3)]
    k, Ts, W = build_svi_surface(maturities, params)
    assert W.shape == (2, len(k))
    assert np.all(W[1] >= W[0] - 1e-12)
