import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pde import (
    heat_btcs,
    heat_crank_nicolson,
    heat_exact_sine,
    heat_fourier_number,
    heat_ftcs,
    l2_error,
    wave_cfl_number,
    wave_exact_sine,
    wave_leapfrog,
)


def test_heat_converges():
    kappa = 0.05
    x, t, U = heat_crank_nicolson(kappa, nx=120, t_final=0.1, nt=100)
    err = l2_error(U[-1], heat_exact_sine(x, t[-1], kappa), x[1] - x[0])
    assert err < 5e-4


def test_wave_cfl_stable_and_accurate():
    c = 1.0
    x, t, U = wave_leapfrog(c, nx=160, t_final=0.4, cfl=0.8)
    err = l2_error(U[-1], wave_exact_sine(x, t[-1], c), x[1] - x[0])
    assert err < 2e-2


def test_heat_boundary_zero():
    x, t, U = heat_crank_nicolson(0.1, nx=50, t_final=0.05, nt=20)
    assert np.allclose(U[:, 0], 0.0)
    assert np.allclose(U[:, -1], 0.0)


def test_ftcs_stable_when_r_le_half():
    """Résultat du rapport : FTCS reste précis si r ≤ 1/2."""
    kappa = 0.1
    nx, nt, t_final = 50, 200, 0.05
    dx = 1.0 / nx
    dt = t_final / nt
    r = heat_fourier_number(kappa, dt, dx)
    assert r <= 0.5
    x, t, U = heat_ftcs(kappa, nx=nx, t_final=t_final, nt=nt)
    err = l2_error(U[-1], heat_exact_sine(x, t[-1], kappa), dx)
    assert err < 1e-2
    assert np.isfinite(U).all()


def test_btcs_unconditionally_stable_large_r():
    """BTCS reste borné même pour r ≫ 1/2 (contrairement à FTCS)."""
    kappa = 0.1
    nx, nt, t_final = 80, 10, 0.2
    dx = 1.0 / nx
    dt = t_final / nt
    r = heat_fourier_number(kappa, dt, dx)
    assert r > 0.5
    x, t, U = heat_btcs(kappa, nx=nx, t_final=t_final, nt=nt)
    assert np.isfinite(U).all()
    assert np.max(np.abs(U)) < 2.0


def test_cn_order_approximately_two():
    """Ordre spatial observé ~ 2 pour Crank–Nicolson (rapport §7)."""
    kappa = 0.1
    errs = []
    for nx in (40, 80):
        x, t, U = heat_crank_nicolson(kappa, nx=nx, t_final=0.1, nt=400)
        err = l2_error(U[-1], heat_exact_sine(x, t[-1], kappa), x[1] - x[0])
        errs.append(err)
    order = np.log(errs[0] / errs[1]) / np.log(2)
    assert 1.5 < order < 2.5


def test_wave_cfl_helper():
    assert abs(wave_cfl_number(1.0, 0.005, 0.01) - 0.5) < 1e-12
