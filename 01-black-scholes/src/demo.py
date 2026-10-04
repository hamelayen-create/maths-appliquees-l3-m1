#!/usr/bin/env python3
"""Démo comparative formule fermée / PDE / Monte Carlo."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bs_closed_form import call_price
from bs_mc import price_call_mc
from bs_pde import price_call_crank_nicolson


def main() -> None:
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    closed = call_price(S0, K, T, r, sigma)
    pde = price_call_crank_nicolson(S0, K, T, r, sigma, n_space=250, n_time=250)
    mc, se = price_call_mc(S0, K, T, r, sigma, n_paths=200_000)

    print("=== Black–Scholes call ATM ===")
    print(f"Paramètres : S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Formule fermée : {closed:.6f}")
    print(f"EDP CN         : {pde:.6f}  (err={pde - closed:+.6f})")
    print(f"Monte Carlo    : {mc:.6f} ± {1.96 * se:.6f} (95%)  (err={mc - closed:+.6f})")


if __name__ == "__main__":
    main()
