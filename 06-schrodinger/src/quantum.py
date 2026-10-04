"""Mécanique quantique 1D numérique."""

from __future__ import annotations

import numpy as np


HBAR = 1.0
MASS = 1.0


def infinite_well_spectrum(n: int, L: float = 1.0) -> float:
    """Énergie analytique E_n = (n² π² ħ²) / (2 m L²), n=1,2,..."""
    return (n**2 * np.pi**2 * HBAR**2) / (2.0 * MASS * L**2)


def hamiltonian_matrix(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Hamiltonien tridiagonal dense H = T + V."""
    dx = x[1] - x[0]
    n = len(x)
    coeff = HBAR**2 / (2.0 * MASS * dx**2)
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        H[i, i] = 2 * coeff + V[i]
        if i > 0:
            H[i, i - 1] = -coeff
        if i < n - 1:
            H[i, i + 1] = -coeff
    return H


def gaussian_packet(
    x: np.ndarray,
    x0: float,
    k0: float,
    sigma: float,
) -> np.ndarray:
    """Paquet d'ondes gaussien normalisé."""
    psi = np.exp(-(x - x0) ** 2 / (2 * sigma**2) + 1j * k0 * x)
    dx = x[1] - x[0]
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi.astype(np.complex128)


def barrier_potential(x: np.ndarray, x_left: float, x_right: float, height: float) -> np.ndarray:
    V = np.zeros_like(x)
    V[(x >= x_left) & (x <= x_right)] = height
    return V


def split_operator_propagate(
    psi: np.ndarray,
    x: np.ndarray,
    V: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """
    Propagation split-operator : e^{-i V dt/2} F^{-1} e^{-i T dt} F e^{-i V dt/2}.
    Retourne psi(t_final).
    """
    dx = x[1] - x[0]
    n = len(x)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    T = (HBAR**2 * k**2) / (2.0 * MASS)
    phase_V_half = np.exp(-1j * V * dt / (2.0 * HBAR))
    phase_T = np.exp(-1j * T * dt / HBAR)

    psi = psi.astype(np.complex128).copy()
    for _ in range(n_steps):
        psi *= phase_V_half
        psi_k = np.fft.fft(psi)
        psi_k *= phase_T
        psi = np.fft.ifft(psi_k)
        psi *= phase_V_half
    return psi


def norm(psi: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx))


def transmission_estimate(psi: np.ndarray, x: np.ndarray, x_cut: float) -> float:
    """Probabilité à droite de x_cut."""
    dx = x[1] - x[0]
    mask = x > x_cut
    return float(np.sum(np.abs(psi[mask]) ** 2) * dx)
