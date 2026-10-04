import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum import (
    gaussian_packet,
    hamiltonian_matrix,
    infinite_well_spectrum,
    norm,
    split_operator_propagate,
)


def test_infinite_well_ground_state():
    L, n = 1.0, 400
    x = np.linspace(0, L, n)
    x_int = x[1:-1]
    H = hamiltonian_matrix(x_int, np.zeros_like(x_int))
    e0 = np.min(np.linalg.eigvalsh(H))
    assert abs(e0 - infinite_well_spectrum(1, L)) / infinite_well_spectrum(1, L) < 0.02


def test_norm_conservation_free_particle():
    x = np.linspace(-30, 30, 1024)
    V = np.zeros_like(x)
    psi0 = gaussian_packet(x, x0=-5.0, k0=3.0, sigma=1.2)
    dx = x[1] - x[0]
    psi = split_operator_propagate(psi0, x, V, dt=0.02, n_steps=100)
    assert abs(norm(psi, dx) - 1.0) < 1e-6
