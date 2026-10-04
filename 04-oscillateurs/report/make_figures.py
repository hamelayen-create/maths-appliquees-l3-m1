#!/usr/bin/env python3
"""Génère les figures du rapport 04-oscillateurs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oscillators import (  # noqa: E402
    damped_harmonic_analytic,
    duffing_poincare,
    fft_spectrum,
    forced_steady_state_analytic,
    quality_factor,
    resonance_curve,
    resonance_peak_frequency,
    simulate_duffing,
    simulate_forced_oscillator,
    simulate_free_oscillator,
    simulate_rk4,
    steady_state_amplitude,
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


def save(fig: plt.Figure, name: str) -> None:
    path = FIGDIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig01_libre_ana_vs_num() -> None:
    omega0, gamma = 2.0 * np.pi, 0.3
    t, x_num, _ = simulate_free_oscillator((0, 8), (1.0, 0.0), omega0, gamma, n_eval=2000)
    x_ana = damped_harmonic_analytic(t, 1.0, 0.0, omega0, gamma)
    err = np.max(np.abs(x_num - x_ana))

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    axes[0].plot(t, x_ana, "k-", lw=2.0, label="Analytique")
    axes[0].plot(t, x_num, "C0--", lw=1.4, label="RK45 (SciPy)")
    axes[0].set_ylabel(r"$x(t)$")
    axes[0].set_title(r"Oscillateur libre sous-amorti : analytique vs numérique"
                      + f"\n($\\omega_0=2\\pi$, $\\gamma=0.3$, $Q={quality_factor(omega0, gamma):.2f}$)")
    axes[0].legend(loc="upper right")

    axes[1].semilogy(t, np.abs(x_num - x_ana) + 1e-16, "C3-", lw=1.2)
    axes[1].axhline(err, color="gray", ls=":", label=f"max = {err:.2e}")
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$|x_{\mathrm{num}}-x_{\mathrm{ana}}|$")
    axes[1].legend(loc="upper right")
    save(fig, "01_libre_ana_vs_num.png")


def fig02_resonance_curve() -> None:
    omega0 = 2.0 * np.pi
    gammas = [0.15, 0.30, 0.60]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for g in gammas:
        w, a = resonance_curve(omega0, g, 1.0, np.linspace(0.2, 2.2 * omega0, 400))
        wr = resonance_peak_frequency(omega0, g)
        ax.plot(w / omega0, a, lw=2, label=rf"$\gamma={g}$, $Q={quality_factor(omega0, g):.1f}$")
        if wr > 0:
            ax.axvline(wr / omega0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel(r"$\omega / \omega_0$")
    ax.set_ylabel(r"$A(\omega)$")
    ax.set_title(r"Courbe de résonance $A(\omega)$ (régime permanent linéaire)")
    ax.legend()
    save(fig, "02_resonance_A_omega.png")


def fig03_duffing_phase() -> None:
    t, x, v = simulate_duffing((0, 120), (0.1, 0.0), n_eval=8000)
    mask = t > 40
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot(x[mask], v[mask], "C0-", lw=0.55, alpha=0.85)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v=\dot x$")
    ax.set_title("Portrait de phase — oscillateur de Duffing (régime permanent)")
    ax.set_aspect("equal", adjustable="datalim")
    save(fig, "03_duffing_phase.png")


def fig04_fft_steady() -> None:
    omega = 1.2
    t, x, _ = simulate_duffing((0, 160), (0.1, 0.0), omega=omega, n_eval=12000)
    mask = t > 60
    freqs, amp = fft_spectrum(t[mask], x[mask])
    f_drive = omega / (2 * np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5))
    axes[0].plot(t[mask], x[mask], "C0-", lw=0.8)
    axes[0].set_ylabel(r"$x(t)$")
    axes[0].set_title("Duffing — signal en régime permanent")
    axes[0].set_xlabel(r"$t$")

    axes[1].plot(freqs, amp, "k-", lw=1.2)
    axes[1].axvline(f_drive, color="C3", ls="--", label=rf"$f_{{\mathrm{{drive}}}}={f_drive:.3f}$ Hz")
    axes[1].set_xlim(0, 1.5)
    axes[1].set_xlabel(r"$f$ (Hz)")
    axes[1].set_ylabel(r"$|X(f)|$")
    axes[1].set_title("Spectre FFT unilatéral (après retrait de la moyenne)")
    axes[1].legend()
    save(fig, "04_fft_regime_permanent.png")


def fig05_bifurcation_qualitative() -> None:
    """Diagramme qualitatif d'amplitude vs force de forçage (balayage γ)."""
    gammas = np.linspace(0.05, 0.55, 28)
    amps_up, amps_down = [], []
    # balayage croissant / décroissant pour hysteresis qualitative
    y0 = np.array([0.1, 0.0])
    for g in gammas:
        t, x, v = simulate_duffing((0, 100), tuple(y0), gamma=float(g), n_eval=5000)
        mask = t > 70
        amps_up.append(0.5 * (np.max(x[mask]) - np.min(x[mask])))
        y0 = np.array([x[-1], v[-1]])
    y0 = np.array([0.1, 0.0])
    for g in gammas[::-1]:
        t, x, v = simulate_duffing((0, 100), tuple(y0), gamma=float(g), n_eval=5000)
        mask = t > 70
        amps_down.append(0.5 * (np.max(x[mask]) - np.min(x[mask])))
        y0 = np.array([x[-1], v[-1]])
    amps_down = amps_down[::-1]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(gammas, amps_up, "C0o-", ms=4, label="Balayage croissant")
    ax.plot(gammas, amps_down, "C3s--", ms=4, label="Balayage décroissant")
    ax.set_xlabel(r"Amplitude de forçage $\gamma$ (notation Duffing)")
    ax.set_ylabel(r"Amplitude de $x$ (régime permanent)")
    ax.set_title("Diagramme qualitatif de bifurcation / hystérésis (Duffing)")
    ax.legend()
    save(fig, "05_bifurcation_duffing.png")


def fig06_schema_modele() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Schéma conceptuel du modèle d'oscillateur forcé", pad=12)

    boxes = [
        (0.4, 1.4, 2.0, 1.4, "Forçage\n$F_0\\cos(\\omega t)$"),
        (3.2, 1.4, 3.4, 1.4, "Système\n$x''+2\\gamma x'+\\omega_0^2 x$"),
        (7.4, 1.4, 2.0, 1.4, "Réponse\n$x(t)$, $A(\\omega)$"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                facecolor="#e8f1fb", edgecolor="#1f4e79", lw=1.8,
            )
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11)

    for x0, x1 in [(2.4, 3.2), (6.6, 7.4)]:
        ax.annotate(
            "",
            xy=(x1, 2.1),
            xytext=(x0, 2.1),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f4e79"),
        )

    ax.text(4.9, 0.55, r"Paramètres : $\omega_0$, $\gamma$, $F_0$, $\omega$", ha="center", fontsize=11)
    ax.text(
        4.9,
        3.55,
        "Linéaire ↔ Duffing : remplacer $\\omega_0^2 x$ par $\\alpha x + \\beta x^3$",
        ha="center",
        fontsize=10,
        style="italic",
    )
    save(fig, "06_schema_modele.png")


def fig07_trois_regimes() -> None:
    omega0 = 2.0
    specs = [
        (0.5, "Sous-amorti", "C0"),
        (2.0, "Critique", "C1"),
        (3.5, "Sur-amorti", "C3"),
    ]
    t = np.linspace(0, 6, 800)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for g, label, c in specs:
        x = damped_harmonic_analytic(t, 1.0, 0.0, omega0, g)
        ax.plot(t, x, color=c, lw=2, label=rf"{label} ($\gamma={g}$)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x(t)$")
    ax.set_title(r"Trois régimes d'amortissement ($\omega_0=2$, $x_0=1$, $v_0=0$)")
    ax.legend()
    save(fig, "07_trois_regimes.png")


def fig08_force_transient() -> None:
    omega0, gamma, F0 = 2.0 * np.pi, 0.35, 1.0
    wd = omega0
    t, x = simulate_forced_oscillator((0, 25), (0.0, 0.0), omega0, gamma, F0, wd, n_eval=4000)
    xp = forced_steady_state_analytic(t, omega0, gamma, F0, wd)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t, x, "C0-", lw=1.3, label="Solution complète (numérique)")
    ax.plot(t, xp, "C3--", lw=1.5, label="Régime permanent analytique")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x(t)$")
    ax.set_title("Oscillateur forcé à la résonance : transitoire → permanent")
    ax.legend()
    save(fig, "08_force_transitoire.png")


def fig09_rk4_vs_rk45() -> None:
    omega0, gamma = 2.0 * np.pi, 0.25

    def f(_t, y):
        x, v = y
        return np.array([v, -2 * gamma * v - omega0**2 * x])

    t_span = (0.0, 6.0)
    n_steps = 400
    t_rk4, y_rk4 = simulate_rk4(f, t_span, np.array([1.0, 0.0]), n_steps)
    x_rk4 = y_rk4[:, 0]
    t_ref = np.linspace(*t_span, n_steps + 1)
    x_ana = damped_harmonic_analytic(t_ref, 1.0, 0.0, omega0, gamma)
    _, x_rk45, _ = simulate_free_oscillator(t_span, (1.0, 0.0), omega0, gamma, n_eval=n_steps + 1)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.semilogy(t_ref, np.abs(x_rk4 - x_ana) + 1e-16, "C0-", label="RK4 pas constant")
    ax.semilogy(t_ref, np.abs(x_rk45 - x_ana) + 1e-16, "C3-", label="RK45 adaptatif (SciPy)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"erreur absolue vs analytique")
    ax.set_title("Comparaison de méthodes numériques (libre sous-amorti)")
    ax.legend()
    save(fig, "09_comparaison_rk4_rk45.png")


def fig10_poincare() -> None:
    omega = 1.2
    t, x, v = simulate_duffing((0, 200), (0.05, 0.0), omega=omega, n_eval=20000)
    xp, vp = duffing_poincare(t, x, v, omega=omega, t_transient=50.0)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot(xp, vp, "k.", ms=3, alpha=0.75)
    ax.set_xlabel(r"$x(t_k)$")
    ax.set_ylabel(r"$v(t_k)$")
    ax.set_title(r"Section de Poincaré stroboscopique (période $2\pi/\omega$)")
    save(fig, "10_poincare_duffing.png")


def main() -> None:
    print("Génération des figures →", FIGDIR)
    fig01_libre_ana_vs_num()
    fig02_resonance_curve()
    fig03_duffing_phase()
    fig04_fft_steady()
    fig05_bifurcation_qualitative()
    fig06_schema_modele()
    fig07_trois_regimes()
    fig08_force_transient()
    fig09_rk4_vs_rk45()
    fig10_poincare()
    print("OK —", len(list(FIGDIR.glob("*.png"))), "PNG")


if __name__ == "__main__":
    main()
