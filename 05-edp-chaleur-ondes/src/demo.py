#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pde import (
    heat_btcs,
    heat_crank_nicolson,
    heat_exact_sine,
    heat_ftcs,
    l2_error,
    wave_exact_sine,
    wave_leapfrog,
)


def main() -> None:
    kappa = 0.1
    x, t, U = heat_crank_nicolson(kappa, nx=100, t_final=0.2, nt=80)
    u_ex = heat_exact_sine(x, t[-1], kappa)
    err_h = l2_error(U[-1], u_ex, x[1] - x[0])
    print("=== EDP chaleur / ondes ===")
    print(f"Chaleur CN : erreur L2 à t={t[-1]:.2f} = {err_h:.3e}")

    # Comparaison FTCS / BTCS / CN (même grille, r < 1/2 pour FTCS)
    nx, nt, t_final = 80, 400, 0.1
    dx = 1.0 / nx
    dt = t_final / nt
    r = kappa * dt / dx**2
    print(f"Comparaison schémas (r = {r:.3f})")
    for name, solver in (
        ("FTCS", heat_ftcs),
        ("BTCS", heat_btcs),
        ("CN", heat_crank_nicolson),
    ):
        x, t, U = solver(kappa, nx=nx, t_final=t_final, nt=nt)
        err = l2_error(U[-1], heat_exact_sine(x, t[-1], kappa), x[1] - x[0])
        print(f"  {name:4s} : erreur L2 = {err:.3e}")

    c = 1.0
    xw, tw, Uw = wave_leapfrog(c, nx=200, t_final=0.5, cfl=0.9)
    u_w_ex = wave_exact_sine(xw, tw[-1], c)
    err_w = l2_error(Uw[-1], u_w_ex, xw[1] - xw[0])
    print(f"Onde leapfrog : erreur L2 à t={tw[-1]:.2f} = {err_w:.3e}")

    # Ordre approximatif chaleur (raffinement x)
    errs = []
    for nx in (40, 80, 160):
        x, t, U = heat_crank_nicolson(kappa, nx=nx, t_final=0.1, nt=200)
        err = l2_error(U[-1], heat_exact_sine(x, t[-1], kappa), x[1] - x[0])
        errs.append(err)
    order = np.log(errs[0] / errs[1]) / np.log(2)
    print(f"Ordre observé chaleur (x) ≈ {order:.2f}")
    print("erreurs =", [f"{e:.3e}" for e in errs])


if __name__ == "__main__":
    main()
