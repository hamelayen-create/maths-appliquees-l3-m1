"""Mécanique quantique 1D numérique.

Discrétisation du Hamiltonien, spectre stationnaire, paquets d'ondes
et propagation split-operator (FFT).
"""

from __future__ import annotations

import numpy as np


HBAR = 1.0
MASS = 1.0


def infinite_well_spectrum(n: int, L: float = 1.0) -> float:
    """Énergie analytique E_n = (n² π² ħ²) / (2 m L²), n=1,2,..."""
    if n < 1:
        raise ValueError("n doit être un entier ≥ 1")
    return (n**2 * np.pi**2 * HBAR**2) / (2.0 * MASS * L**2)


def infinite_well_wavefunction(x: np.ndarray, n: int, L: float = 1.0) -> np.ndarray:
    """Fonction propre analytique du puits infini [0, L] : ψ_n(x) = √(2/L) sin(n π x / L)."""
    if n < 1:
        raise ValueError("n doit être un entier ≥ 1")
    psi = np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)
    psi = np.where((x > 0.0) & (x < L), psi, 0.0)
    return psi.astype(float)


def hamiltonian_matrix(x: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Hamiltonien tridiagonal dense H = T + V (différences finies centrées)."""
    dx = float(x[1] - x[0])
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


def eigenpairs(x: np.ndarray, V: np.ndarray, k: int | None = None):
    """
    Diagonalise H = T+V.

    Retourne (energies, vectors) triés par énergie croissante.
    vectors[:, j] est le j-ième état propre (non normalisé en L² dx si besoin).
    """
    H = hamiltonian_matrix(x, V)
    energies, vectors = np.linalg.eigh(H)
    order = np.argsort(energies)
    energies = energies[order]
    vectors = vectors[:, order]
    dx = float(x[1] - x[0])
    for j in range(vectors.shape[1]):
        nrm = np.sqrt(np.sum(np.abs(vectors[:, j]) ** 2) * dx)
        if nrm > 0:
            vectors[:, j] /= nrm
        # phase convention: première composante dominante ≥ 0
        idx = int(np.argmax(np.abs(vectors[:, j])))
        if vectors[idx, j] < 0:
            vectors[:, j] *= -1
    if k is not None:
        return energies[:k], vectors[:, :k]
    return energies, vectors


def gaussian_packet(
    x: np.ndarray,
    x0: float,
    k0: float,
    sigma: float,
) -> np.ndarray:
    """Paquet d'ondes gaussien normalisé (∥ψ∥₂ = 1)."""
    psi = np.exp(-(x - x0) ** 2 / (2 * sigma**2) + 1j * k0 * x)
    dx = float(x[1] - x[0])
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi.astype(np.complex128)


def barrier_potential(
    x: np.ndarray,
    x_left: float,
    x_right: float,
    height: float,
) -> np.ndarray:
    """Potentiel barrière rectangulaire de hauteur `height` sur [x_left, x_right]."""
    V = np.zeros_like(x, dtype=float)
    V[(x >= x_left) & (x <= x_right)] = height
    return V


def harmonic_potential(x: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """Oscillateur harmonique V(x) = (1/2) m ω² x²."""
    return 0.5 * MASS * omega**2 * x**2


def harmonic_spectrum(n: int, omega: float = 1.0) -> float:
    """Énergie analytique E_n = ħ ω (n + 1/2), n=0,1,2,..."""
    if n < 0:
        raise ValueError("n doit être un entier ≥ 0")
    return HBAR * omega * (n + 0.5)


def split_operator_propagate(
    psi: np.ndarray,
    x: np.ndarray,
    V: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """
    Propagation split-operator (Strang) :
    e^{-i V dt/2} F^{-1} e^{-i T dt} F e^{-i V dt/2}.
    Retourne psi(t_final).
    """
    history = split_operator_history(psi, x, V, dt, n_steps, save_every=n_steps)
    return history[-1]


def split_operator_step(
    psi: np.ndarray,
    x: np.ndarray,
    V: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Un seul pas de split-operator (alias pédagogique)."""
    return split_operator_propagate(psi, x, V, dt, n_steps=1)


def split_operator_history(
    psi: np.ndarray,
    x: np.ndarray,
    V: np.ndarray,
    dt: float,
    n_steps: int,
    save_every: int = 1,
) -> list[np.ndarray]:
    """
    Propagation split-operator avec historique.

    Retourne une liste de copies de ψ, y compris l'état initial, sauvegardées
    toutes les `save_every` itérations (et toujours à la fin).
    """
    if save_every < 1:
        raise ValueError("save_every doit être ≥ 1")
    dx = float(x[1] - x[0])
    n = len(x)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    T = (HBAR**2 * k**2) / (2.0 * MASS)
    phase_V_half = np.exp(-1j * V * dt / (2.0 * HBAR))
    phase_T = np.exp(-1j * T * dt / HBAR)

    psi = psi.astype(np.complex128).copy()
    out = [psi.copy()]
    for step in range(1, n_steps + 1):
        psi *= phase_V_half
        psi_k = np.fft.fft(psi)
        psi_k *= phase_T
        psi = np.fft.ifft(psi_k)
        psi *= phase_V_half
        if step % save_every == 0 or step == n_steps:
            out.append(psi.copy())
    return out


def norm(psi: np.ndarray, dx: float) -> float:
    """Norme L² discrète √(Σ |ψ|² Δx)."""
    return float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx))


def probability_density(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


def transmission_estimate(psi: np.ndarray, x: np.ndarray, x_cut: float) -> float:
    """Probabilité à droite de x_cut (estimateur de transmission)."""
    dx = float(x[1] - x[0])
    mask = x > x_cut
    return float(np.sum(np.abs(psi[mask]) ** 2) * dx)


def reflection_estimate(psi: np.ndarray, x: np.ndarray, x_cut: float) -> float:
    """Probabilité à gauche de x_cut (estimateur de réflexion)."""
    dx = float(x[1] - x[0])
    mask = x < x_cut
    return float(np.sum(np.abs(psi[mask]) ** 2) * dx)
