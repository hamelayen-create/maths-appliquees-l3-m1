#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio import (
    efficient_frontier_analytic,
    optimize_long_only,
    portfolio_stats,
    simulate_returns,
)


def main() -> None:
    R, mu, cov = simulate_returns()
    print("=== Markowitz (données synthétiques annualisées) ===")
    print("mu =", np.round(mu, 4))
    print("vol =", np.round(np.sqrt(np.diag(cov)), 4))

    targets, vols, _ = efficient_frontier_analytic(mu, cov, n_points=15)
    print("\nFrontière analytique (extrait) :")
    for t, v in zip(targets[::3], vols[::3]):
        print(f"  target={t:.3f}  vol={v:.3f}")

    target = float(np.median(mu))
    w = optimize_long_only(mu, cov, target_return=target)
    ret, vol = portfolio_stats(w, mu, cov)
    print(f"\nLong-only @ target≈{target:.3f}")
    print("weights =", np.round(w, 4))
    print(f"realized return={ret:.4f}, vol={vol:.4f}, Sharpe≈{ret/vol:.3f}")


if __name__ == "__main__":
    main()
