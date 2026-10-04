#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantum import (
    barrier_potential,
    eigenpairs,
    gaussian_packet,
    infinite_well_spectrum,
    norm,
    split_operator_propagate,
    transmission_estimate,
)


def main() -> None:
    print("=== Schrödinger 1D ===")
    # Spectre puits infini (Dirichlet via points intérieurs)
    L = 1.0
    n = 300
    x = np.linspace(0, L, n)
    x_int = x[1:-1]
    energies, _ = eigenpairs(x_int, np.zeros_like(x_int), k=3)
    for k in range(1, 4):
        e_th = infinite_well_spectrum(k, L=L)
        print(f"E_{k}: num={energies[k-1]:.6f}, th={e_th:.6f}, err={abs(energies[k-1]-e_th):.3e}")

    # Tunnellisation
    x = np.linspace(-20, 20, 1024)
    V = barrier_potential(x, 0.0, 1.5, height=8.0)
    psi0 = gaussian_packet(x, x0=-8.0, k0=4.0, sigma=1.0)
    dx = x[1] - x[0]
    print(f"norme initiale = {norm(psi0, dx):.10f}")
    psi = split_operator_propagate(psi0, x, V, dt=0.01, n_steps=400)
    print(f"norme finale   = {norm(psi, dx):.10f}")
    T = transmission_estimate(psi, x, x_cut=1.5)
    print(f"probabilité transmise (x>1.5) ≈ {T:.4f}")


if __name__ == "__main__":
    main()
