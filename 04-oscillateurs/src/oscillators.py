"""Oscillateurs linéaires / Duffing, résonance et FFT.

Référence pédagogique pour le dossier ``04-oscillateurs``.
Convention d'amortissement : l'équation libre s'écrit

    x'' + 2 γ x' + ω0² x = 0

avec γ > 0 (taux d'amortissement) et ω0 > 0 (pulsation propre).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def damping_regime(omega0: float, gamma: float) -> str:
    """Retourne ``'sous'``, ``'critique'`` ou ``'sur'`` selon le discriminant."""
    disc = gamma**2 - omega0**2
    if abs(disc) < 1e-14 * max(omega0**2, 1.0):
        return "critique"
    return "sous" if disc < 0 else "sur"


def quality_factor(omega0: float, gamma: float) -> float:
    """Facteur de qualité Q = ω0 / (2 γ) pour l'amortissement 2γ."""
    if gamma <= 0:
        raise ValueError("gamma doit être strictement positif")
    return omega0 / (2.0 * gamma)


def damped_harmonic_analytic(
    t: np.ndarray,
    x0: float,
    v0: float,
    omega0: float,
    gamma: float,
) -> np.ndarray:
    """Solution libre exacte : x'' + 2γ x' + ω0² x = 0 (trois régimes)."""
    t = np.asarray(t, dtype=float)
    regime = damping_regime(omega0, gamma)

    if regime == "sous":
        omega = np.sqrt(omega0**2 - gamma**2)
        A = x0
        B = (v0 + gamma * x0) / omega
        return np.exp(-gamma * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))

    if regime == "critique":
        # x(t) = e^{-γ t} (A + B t), γ = ω0
        A = x0
        B = v0 + gamma * x0
        return np.exp(-gamma * t) * (A + B * t)

    # sur-amorti : racines -γ ± μ, μ = sqrt(γ² - ω0²)
    mu = np.sqrt(gamma**2 - omega0**2)
    # x = e^{-γt} (C cosh μt + D sinh μt)
    C = x0
    D = (v0 + gamma * x0) / mu
    return np.exp(-gamma * t) * (C * np.cosh(mu * t) + D * np.sinh(mu * t))


def simulate_free_oscillator(
    t_span: tuple[float, float],
    y0: tuple[float, float],
    omega0: float,
    gamma: float,
    n_eval: int = 2000,
    method: str = "RK45",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intégration numérique libre : retourne t, x, v."""

    def f(_t, y):
        x, v = y
        return [v, -2 * gamma * v - omega0**2 * x]

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(f, t_span, y0, t_eval=t_eval, method=method, rtol=1e-8, atol=1e-8)
    return sol.t, sol.y[0], sol.y[1]


def rk4_step(f, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """Un pas Runge–Kutta d'ordre 4 explicite."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_rk4(
    f,
    t_span: tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Intègre y' = f(t, y) par RK4 à pas constant."""
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    y = np.zeros((n_steps + 1, len(y0)), dtype=float)
    y[0] = np.asarray(y0, dtype=float)
    for i in range(n_steps):
        y[i + 1] = rk4_step(f, t[i], y[i], h)
    return t, y


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


def forced_steady_state_analytic(
    t: np.ndarray,
    omega0: float,
    gamma: float,
    F0: float,
    omega_drive: float,
) -> np.ndarray:
    """Partie particulière (régime permanent) x_p = D cos(ωt − φ)."""
    t = np.asarray(t, dtype=float)
    w = omega_drive
    denom = (omega0**2 - w**2) ** 2 + (2 * gamma * w) ** 2
    D = F0 / np.sqrt(denom)
    # cos φ = (ω0² − ω²)/√denom, sin φ = 2γω/√denom
    cos_phi = (omega0**2 - w**2) / np.sqrt(denom)
    sin_phi = (2 * gamma * w) / np.sqrt(denom)
    phi = np.arctan2(sin_phi, cos_phi)
    return D * np.cos(w * t - phi)


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


def resonance_peak_frequency(omega0: float, gamma: float) -> float:
    """Pulsation de résonance en amplitude : √(ω0² − 2γ²) si ω0 > √2 γ, sinon 0."""
    arg = omega0**2 - 2 * gamma**2
    return float(np.sqrt(arg)) if arg > 0 else 0.0


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


def duffing_poincare(
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    omega: float,
    t_transient: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Section de Poincaré stroboscopique à la période du forçage 2π/ω."""
    period = 2.0 * np.pi / omega
    mask = t >= t_transient
    t_ss, x_ss, v_ss = t[mask], x[mask], v[mask]
    if len(t_ss) < 2:
        return np.array([]), np.array([])
    # indices les plus proches de t0 + k T
    t0 = t_ss[0]
    n_periods = int((t_ss[-1] - t0) / period)
    xs, vs = [], []
    for k in range(n_periods):
        target = t0 + k * period
        idx = int(np.argmin(np.abs(t_ss - target)))
        xs.append(x_ss[idx])
        vs.append(v_ss[idx])
    return np.asarray(xs), np.asarray(vs)
