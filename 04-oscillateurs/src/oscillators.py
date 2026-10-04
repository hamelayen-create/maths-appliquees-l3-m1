"""Oscillateurs linéaires / Duffing, résonance et FFT."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def damped_harmonic_analytic(
    t: np.ndarray,
    x0: float,
    v0: float,
    omega0: float,
    gamma: float,
) -> np.ndarray:
    """Solution libre sous-amortie : x'' + 2γ x' + ω0² x = 0."""
    t = np.asarray(t, dtype=float)
    omega = np.sqrt(omega0**2 - gamma**2)
    # x = e^{-γt} (A cos ωt + B sin ωt)
    A = x0
    B = (v0 + gamma * x0) / omega
    return np.exp(-gamma * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))


def simulate_forced_oscillator(
    t_span: tuple[float, float],
    y0: tuple[float, float],
    omega0: float,
    gamma: float,
    F0: float,
    omega_drive: float,
    n_eval: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """x'' + 2γ x' + ω0² x = F0 cos(ω t)."""

    def f(t, y):
        x, v = y
        return [v, -2 * gamma * v - omega0**2 * x + F0 * np.cos(omega_drive * t)]

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(f, t_span, y0, t_eval=t_eval, rtol=1e-8, atol=1e-8)
    return sol.t, sol.y[0]


def simulate_duffing(
    t_span: tuple[float, float],
    y0: tuple[float, float],
    delta: float = 0.2,
    alpha: float = -1.0,
    beta: float = 1.0,
    gamma: float = 0.3,
    omega: float = 1.2,
    n_eval: int = 4000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Duffing : x'' + δ x' + α x + β x³ = γ cos(ω t).
    Retourne t, x, v.
    """

    def f(t, y):
        x, v = y
        return [v, -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)]

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(f, t_span, y0, t_eval=t_eval, rtol=1e-7, atol=1e-7)
    return sol.t, sol.y[0], sol.y[1]


def steady_state_amplitude(
    omega0: float,
    gamma: float,
    F0: float,
    omega_drive: float,
) -> float:
    """Amplitude analytique du régime permanent (linéaire forcé)."""
    denom = (omega0**2 - omega_drive**2) ** 2 + (2 * gamma * omega_drive) ** 2
    return F0 / np.sqrt(denom)


def resonance_curve(
    omega0: float,
    gamma: float,
    F0: float,
    omega_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if omega_grid is None:
        omega_grid = np.linspace(0.1 * omega0, 2.5 * omega0, 200)
    amps = np.array([steady_state_amplitude(omega0, gamma, F0, w) for w in omega_grid])
    return omega_grid, amps


def fft_spectrum(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spectre unilatéral |X(f)| après retrait de la moyenne."""
    dt = t[1] - t[0]
    sig = x - np.mean(x)
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(np.fft.rfft(sig)) * 2.0 / n
    return freqs, amp
