import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantum import (
    barrier_potential,
    eigenpairs,
    gaussian_packet,
    hamiltonian_matrix,
    infinite_well_spectrum,
    infinite_well_wavefunction,
    norm,
    split_operator_history,
    split_operator_propagate,
    transmission_estimate,
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


def test_eigenfunction_overlaps_ground_state():
    """Résultat du rapport : l'état fondamental numérique chevauche ψ₁ analytique."""
    L, n = 1.0, 500
    x = np.linspace(0, L, n)
    x_int = x[1:-1]
    energies, vectors = eigenpairs(x_int, np.zeros_like(x_int), k=1)
    psi_num = vectors[:, 0]
    psi_th = infinite_well_wavefunction(x_int, 1, L=L)
    dx = x_int[1] - x_int[0]
    overlap = abs(np.sum(np.conj(psi_num) * psi_th) * dx)
    assert overlap > 0.99
    assert abs(energies[0] - infinite_well_spectrum(1, L)) / infinite_well_spectrum(1, L) < 0.02


def test_tunneling_transmission_increases_with_energy():
    """Résultat du rapport : à barrière fixe, T croît avec l'impulsion k0."""
    x = np.linspace(-25, 25, 1024)
    V = barrier_potential(x, 0.0, 1.2, height=10.0)
    dx = x[1] - x[0]
    transmissions = []
    for k0 in (2.0, 5.0):
        psi0 = gaussian_packet(x, x0=-8.0, k0=k0, sigma=1.0)
        # énergie cinétique ≈ k0²/2 ; pour k0=2 << barrière, pour k0=5 plus proche
        hist = split_operator_history(psi0, x, V, dt=0.01, n_steps=500, save_every=500)
        psi = hist[-1]
        assert abs(norm(psi, dx) - 1.0) < 1e-5
        transmissions.append(transmission_estimate(psi, x, x_cut=1.2))
    assert transmissions[1] > transmissions[0]
