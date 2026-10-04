#!/usr/bin/env python3
"""Génère les figures du rapport Schrödinger 1D (≥ 8 PNG)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum import (  # noqa: E402
    barrier_potential,
    eigenpairs,
    gaussian_packet,
    hamiltonian_matrix,
    infinite_well_spectrum,
    infinite_well_wavefunction,
    norm,
    split_operator_history,
    transmission_estimate,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def save(fig: plt.Figure, name: str) -> Path:
    path = FIGDIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def fig01_concept_diagram() -> None:
    """Schéma conceptuel du pipeline numérique."""
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.1, "Potentiel\nV(x)"),
        (2.4, 1.1, "Hamiltonien\nH = T + V"),
        (4.6, 1.1, "Spectre\nHψ = Eψ"),
        (6.8, 1.1, "Paquet ψ(x,0)\n+ split-FFT"),
    ]
    for x, y, text in boxes:
        rect = plt.Rectangle((x, y), 1.8, 1.2, fill=True, facecolor="#e8f1f8", edgecolor="#1f4e79", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.9, y + 0.6, text, ha="center", va="center", fontsize=10)
    for x0 in (2.1, 4.3, 6.5):
        ax.annotate("", xy=(x0 + 0.25, 1.7), xytext=(x0 - 0.05, 1.7), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))
    ax.text(5, 0.45, "Conservation ‖ψ‖₂  ·  transmission / réflexion  ·  comparaison analytique", ha="center", fontsize=10)
    ax.set_title("Pipeline : du potentiel au paquet d'ondes (modèle 1D)")
    save(fig, "fig01_schema_conceptuel.png")


def fig02_well_potential_densities() -> None:
    L, n = 1.0, 500
    x = np.linspace(0, L, n)
    x_int = x[1:-1]
    energies, vectors = eigenpairs(x_int, np.zeros_like(x_int), k=3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Potentiel puits (illustration)
    xp = np.linspace(-0.1, 1.1, 400)
    Vp = np.zeros_like(xp)
    Vp[(xp <= 0) | (xp >= L)] = 40.0
    axes[0].plot(xp, Vp, color="#1f4e79", lw=2)
    axes[0].set_ylim(-2, 45)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("V(x)")
    axes[0].set_title("Puits infini (illustration V→∞ hors [0,L])")

    colors = ["#c0392b", "#2980b9", "#27ae60"]
    for j in range(3):
        dens = np.abs(vectors[:, j]) ** 2
        axes[1].plot(x_int, dens + 8 * j, color=colors[j], lw=1.8, label=rf"$|\psi_{j+1}|^2$ (+offset)")
        axes[1].plot(x_int, infinite_well_wavefunction(x_int, j + 1, L) ** 2 + 8 * j, "--", color="gray", alpha=0.7, lw=1)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$|\psi_n(x)|^2$ (décalées)")
    axes[1].set_title("Densités propres numériques vs analytiques")
    axes[1].legend(fontsize=8)
    save(fig, "fig02_puits_potentiel_densites.png")


def fig03_spectrum_vs_analytic() -> None:
    L = 1.0
    ns = [100, 200, 400]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n_modes = 6
    n_grid = 400
    x = np.linspace(0, L, n_grid)
    x_int = x[1:-1]
    e_num = np.sort(np.linalg.eigvalsh(hamiltonian_matrix(x_int, np.zeros_like(x_int))))[:n_modes]
    e_th = np.array([infinite_well_spectrum(k, L) for k in range(1, n_modes + 1)])
    idx = np.arange(1, n_modes + 1)
    axes[0].plot(idx, e_th, "o-", label="analytique", color="#1f4e79")
    axes[0].plot(idx, e_num, "s--", label="numérique", color="#c0392b")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel(r"$E_n$")
    axes[0].set_title("Spectre puits infini (n=1…6)")
    axes[0].legend()

    errs = []
    for n in ns:
        x = np.linspace(0, L, n)
        x_int = x[1:-1]
        e0 = np.min(np.linalg.eigvalsh(hamiltonian_matrix(x_int, np.zeros_like(x_int))))
        errs.append(abs(e0 - infinite_well_spectrum(1, L)) / infinite_well_spectrum(1, L))
    axes[1].loglog(ns, errs, "o-", color="#2980b9")
    axes[1].set_xlabel("nombre de points de grille")
    axes[1].set_ylabel(r"erreur relative sur $E_1$")
    axes[1].set_title("Convergence de l'état fondamental")
    save(fig, "fig03_spectre_vs_analytique.png")


def fig04_packet_snapshots() -> None:
    x = np.linspace(-20, 20, 1024)
    V = barrier_potential(x, 0.0, 1.5, height=8.0)
    psi0 = gaussian_packet(x, x0=-8.0, k0=4.0, sigma=1.0)
    dt, n_steps = 0.01, 400
    save_every = 80
    hist = split_operator_history(psi0, x, V, dt, n_steps, save_every=save_every)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    axes = axes.ravel()
    vmax = max(np.max(np.abs(psi) ** 2) for psi in hist)
    for i, psi in enumerate(hist[:6]):
        t = i * save_every * dt
        axes[i].plot(x, np.abs(psi) ** 2, color="#1f4e79", lw=1.5)
        axes[i].axvspan(0.0, 1.5, color="#f5cba7", alpha=0.5, label="barrière" if i == 0 else None)
        axes[i].set_ylim(0, 1.15 * vmax)
        axes[i].set_title(f"t = {t:.2f}")
        if i >= 3:
            axes[i].set_xlabel("x")
        axes[i].set_ylabel(r"$|\psi|^2$")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Snapshots de tunnellisation (split-operator)")
    save(fig, "fig04_snapshots_tunnel.png")


def fig05_norm_vs_time() -> None:
    x = np.linspace(-25, 25, 1024)
    dx = x[1] - x[0]
    V = barrier_potential(x, 0.0, 1.5, height=8.0)
    psi0 = gaussian_packet(x, x0=-8.0, k0=4.0, sigma=1.0)
    dt, n_steps, save_every = 0.01, 400, 10
    hist = split_operator_history(psi0, x, V, dt, n_steps, save_every=save_every)
    times = np.arange(len(hist)) * save_every * dt
    # corriger le dernier temps si n_steps non multiple
    if (n_steps % save_every) != 0:
        times[-1] = n_steps * dt
    norms = [norm(psi, dx) for psi in hist]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(times, norms, color="#27ae60", lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|\psi(\cdot,t)\|_2$")
    ax.set_title("Conservation de la norme (barrière, split-operator)")
    ax.set_ylim(0.999999, 1.000001)
    save(fig, "fig05_norme_temps.png")


def fig06_transmission_vs_energy() -> None:
    x = np.linspace(-25, 25, 1024)
    V = barrier_potential(x, 0.0, 1.2, height=10.0)
    k0s = np.linspace(1.5, 6.5, 9)
    Ts = []
    for k0 in k0s:
        psi0 = gaussian_packet(x, x0=-8.0, k0=float(k0), sigma=1.0)
        # temps suffisant pour que le paquet interagisse puis se sépare
        psi = split_operator_history(psi0, x, V, dt=0.01, n_steps=550, save_every=550)[-1]
        Ts.append(transmission_estimate(psi, x, x_cut=1.2))
    Ekin = 0.5 * k0s**2  # ħ=m=1

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(Ekin, Ts, "o-", color="#8e44ad", lw=2)
    ax.axvline(10.0, color="#c0392b", ls="--", label="hauteur barrière V0=10")
    ax.set_xlabel(r"énergie cinétique $E \approx k_0^2/2$")
    ax.set_ylabel("probabilité transmise T")
    ax.set_title("Transmission vs énergie (largeur barrière = 1.2)")
    ax.legend()
    save(fig, "fig06_transmission_vs_energie.png")


def fig07_transmission_vs_width() -> None:
    x = np.linspace(-25, 25, 1024)
    widths = np.linspace(0.4, 2.4, 8)
    Ts = []
    for w in widths:
        V = barrier_potential(x, 0.0, float(w), height=10.0)
        psi0 = gaussian_packet(x, x0=-8.0, k0=4.0, sigma=1.0)
        psi = split_operator_history(psi0, x, V, dt=0.01, n_steps=550, save_every=550)[-1]
        Ts.append(transmission_estimate(psi, x, x_cut=float(w)))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.semilogy(widths, np.clip(Ts, 1e-8, 1), "o-", color="#d35400", lw=2)
    ax.set_xlabel("largeur de barrière w")
    ax.set_ylabel("T (échelle log)")
    ax.set_title(r"Transmission vs largeur (k0=4, V0=10)")
    save(fig, "fig07_transmission_vs_largeur.png")


def fig08_free_packet_spreading() -> None:
    x = np.linspace(-40, 40, 2048)
    V = np.zeros_like(x)
    psi0 = gaussian_packet(x, x0=0.0, k0=0.0, sigma=1.5)
    hist = split_operator_history(psi0, x, V, dt=0.02, n_steps=300, save_every=100)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, psi in enumerate(hist):
        t = i * 100 * 0.02
        ax.plot(x, np.abs(psi) ** 2, lw=1.6, label=f"t={t:.1f}")
    ax.set_xlim(-25, 25)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi|^2$")
    ax.set_title("Étalement d'un paquet gaussien libre (k0=0)")
    ax.legend()
    save(fig, "fig08_etalement_paquet_libre.png")


def fig09_error_modes() -> None:
    """Erreur relative E_n pour plusieurs niveaux."""
    L, n = 1.0, 300
    x = np.linspace(0, L, n)
    x_int = x[1:-1]
    e_num = np.sort(np.linalg.eigvalsh(hamiltonian_matrix(x_int, np.zeros_like(x_int))))[:8]
    e_th = np.array([infinite_well_spectrum(k, L) for k in range(1, 9)])
    rel = np.abs(e_num - e_th) / e_th

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.semilogy(np.arange(1, 9), rel, "o-", color="#1f4e79")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$|E_n^{num}-E_n^{th}|/E_n^{th}$")
    ax.set_title("Erreur spectrale relative (grille N=300)")
    save(fig, "fig09_erreur_modes.png")


def main() -> None:
    fig01_concept_diagram()
    fig02_well_potential_densities()
    fig03_spectrum_vs_analytic()
    fig04_packet_snapshots()
    fig05_norm_vs_time()
    fig06_transmission_vs_energy()
    fig07_transmission_vs_width()
    fig08_free_packet_spreading()
    fig09_error_modes()
    print(f"Done. Figures in {FIGDIR}")


if __name__ == "__main__":
    main()
