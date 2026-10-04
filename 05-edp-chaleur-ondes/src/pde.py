"""Schémas différences finies pour chaleur et ondes 1D."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def heat_exact_sine(x: np.ndarray, t: float, kappa: float, L: float = 1.0) -> np.ndarray:
    """Solution exacte u=sin(πx/L) exp(-κ π² t / L²)."""
    return np.sin(np.pi * x / L) * np.exp(-kappa * (np.pi / L) ** 2 * t)


def wave_exact_sine(x: np.ndarray, t: float, c: float, L: float = 1.0) -> np.ndarray:
    """u = sin(πx/L) cos(c π t / L), vitesse initiale nulle."""
    return np.sin(np.pi * x / L) * np.cos(c * np.pi * t / L)


def _heat_init(
    x: np.ndarray,
    kappa: float,
    L: float,
    u0: np.ndarray | None,
) -> np.ndarray:
    if u0 is None:
        return heat_exact_sine(x, 0.0, kappa, L)
    return np.asarray(u0, dtype=float).copy()


def heat_ftcs(
    kappa: float,
    L: float = 1.0,
    nx: int = 100,
    t_final: float = 0.1,
    nt: int = 100,
    u0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FTCS (Forward-Time Central-Space) Dirichlet homogène.
    Stable si r = κ Δt / Δx² ≤ 1/2.
    Retourne x, t_grid, U[nt+1, nx+1].
    """
    x = np.linspace(0.0, L, nx + 1)
    dx = x[1] - x[0]
    dt = t_final / nt
    t_grid = np.linspace(0.0, t_final, nt + 1)
    u = _heat_init(x, kappa, L, u0)
    r = kappa * dt / dx**2

    U = np.zeros((nt + 1, nx + 1))
    U[0] = u
    for n in range(nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = u_new[-1] = 0.0
        u = u_new
        U[n + 1] = u
    return x, t_grid, U


def heat_btcs(
    kappa: float,
    L: float = 1.0,
    nx: int = 100,
    t_final: float = 0.1,
    nt: int = 100,
    u0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BTCS (Backward-Time Central-Space / Euler implicite) Dirichlet homogène.
    Inconditionnellement stable. Retourne x, t_grid, U[nt+1, nx+1].
    """
    x = np.linspace(0.0, L, nx + 1)
    dx = x[1] - x[0]
    dt = t_final / nt
    t_grid = np.linspace(0.0, t_final, nt + 1)
    u = _heat_init(x, kappa, L, u0)
    r = kappa * dt / dx**2

    n_int = nx - 1
    ab = np.zeros((3, n_int))
    ab[0, 1:] = -r
    ab[1, :] = 1 + 2 * r
    ab[2, :-1] = -r

    U = np.zeros((nt + 1, nx + 1))
    U[0] = u
    for n in range(nt):
        rhs = u[1:-1].copy()
        u_new_int = solve_banded((1, 1), ab, rhs)
        u = np.zeros_like(u)
        u[1:-1] = u_new_int
        U[n + 1] = u
    return x, t_grid, U


def heat_crank_nicolson(
    kappa: float,
    L: float = 1.0,
    nx: int = 100,
    t_final: float = 0.1,
    nt: int = 100,
    u0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crank–Nicolson Dirichlet homogène.
    Retourne x, t_grid, U[nt+1, nx+1].
    """
    x = np.linspace(0.0, L, nx + 1)
    dx = x[1] - x[0]
    dt = t_final / nt
    t_grid = np.linspace(0.0, t_final, nt + 1)
    u = _heat_init(x, kappa, L, u0)

    r = kappa * dt / (2 * dx**2)
    n_int = nx - 1
    ab = np.zeros((3, n_int))
    ab[0, 1:] = -r
    ab[1, :] = 1 + 2 * r
    ab[2, :-1] = -r

    U = np.zeros((nt + 1, nx + 1))
    U[0] = u
    for n in range(nt):
        u_int = u[1:-1]
        rhs = r * u[:-2] + (1 - 2 * r) * u_int + r * u[2:]
        u_new_int = solve_banded((1, 1), ab, rhs)
        u = np.zeros_like(u)
        u[1:-1] = u_new_int
        U[n + 1] = u
    return x, t_grid, U


def wave_leapfrog(
    c: float,
    L: float = 1.0,
    nx: int = 200,
    t_final: float = 1.0,
    cfl: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Schéma saute-mouton explicite. CFL = c dt/dx ≤ 1.
    """
    x = np.linspace(0.0, L, nx + 1)
    dx = x[1] - x[0]
    dt = cfl * dx / c
    nt = int(np.ceil(t_final / dt))
    dt = t_final / nt
    lam2 = (c * dt / dx) ** 2
    t_grid = np.linspace(0.0, t_final, nt + 1)

    u_prev = wave_exact_sine(x, 0.0, c, L)
    # étape 1 via Taylor : u1 = u0 + 0.5 dt² u_tt = u0 + 0.5 lam2 * Dxx u0
    u = u_prev.copy()
    u[1:-1] = u_prev[1:-1] + 0.5 * lam2 * (u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2])
    u[0] = u[-1] = 0.0

    U = np.zeros((nt + 1, nx + 1))
    U[0] = u_prev
    if nt >= 1:
        U[1] = u

    for n in range(1, nt):
        u_next = np.zeros_like(u)
        u_next[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + lam2 * (
            u[2:] - 2 * u[1:-1] + u[:-2]
        )
        u_prev, u = u, u_next
        U[n + 1] = u
    return x, t_grid, U


def l2_error(u_num: np.ndarray, u_ex: np.ndarray, dx: float) -> float:
    return float(np.sqrt(dx * np.sum((u_num - u_ex) ** 2)))


def heat_fourier_number(kappa: float, dt: float, dx: float) -> float:
    """Nombre de Fourier r = κ Δt / Δx² (critère FTCS : r ≤ 1/2)."""
    return kappa * dt / dx**2


def wave_cfl_number(c: float, dt: float, dx: float) -> float:
    """Nombre CFL λ = c Δt / Δx (critère leapfrog : λ ≤ 1)."""
    return c * dt / dx
