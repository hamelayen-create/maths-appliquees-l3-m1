import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pde import heat_crank_nicolson, heat_exact_sine, l2_error, wave_exact_sine, wave_leapfrog


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
