#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from implied_vol import implied_vol_call
from svi import calibrate_svi, make_synthetic_smile, svi_total_variance


def main() -> None:
    S, r, T = 100.0, 0.01, 0.5
    F = S * np.exp(r * T)
    true = (0.03, 0.25, -0.35, 0.05, 0.25)
    K, iv_true, w_true = make_synthetic_smile(F=F, T=T, true_params=true, noise=1e-4)

    # Round-trip prix → implied vol
    from scipy.stats import norm

    def bs_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    iv_rec = np.array([
        implied_vol_call(float(bs_call(S, k, T, r, sig)), S, float(k), T, r)
        for k, sig in zip(K, iv_true)
    ])
    print("=== Smile / SVI ===")
    print("max |iv_rec - iv_true| =", float(np.max(np.abs(iv_rec - iv_true))))

    k = np.log(K / F)
    fit = calibrate_svi(k, w_true)
    print("SVI fit:", {key: round(fit[key], 5) if isinstance(fit[key], float) else fit[key]
                       for key in ("a", "b", "rho", "m", "sigma", "rmse", "success")})
    w_fit = svi_total_variance(k, fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"])
    print("RMSE total variance =", float(np.sqrt(np.mean((w_fit - w_true) ** 2))))


if __name__ == "__main__":
    main()
