import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from implied_vol import implied_vol_call
from svi import calibrate_svi, make_synthetic_smile, svi_total_variance


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
