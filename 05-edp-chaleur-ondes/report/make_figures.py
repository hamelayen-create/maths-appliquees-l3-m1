#!/usr/bin/env python3
"""Génère les figures du rapport EDP chaleur / ondes."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pde import (  # noqa: E402
    heat_btcs,
    heat_crank_nicolson,
    heat_exact_sine,
    heat_ftcs,
    l2_error,
    wave_exact_sine,
    wave_leapfrog,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def fig01_heat_evolution() -> None:
    """Évolution u(x,t) pour l'équation de la chaleur (CN)."""
    kappa = 0.1
    x, t, U = heat_crank_nicolson(kappa, nx=120, t_final=0.4, nt=160)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    times = [0.0, 0.05, 0.1, 0.2, 0.4]
    for ti in times:
        idx = int(np.argmin(np.abs(t - ti)))
        axes[0].plot(x, U[idx], label=f"t = {t[idx]:.2f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x,t)")
    axes[0].set_title("Profils spatiaux (Crank–Nicolson)")
    axes[0].legend(fontsize=9)

    T, X = np.meshgrid(t, x, indexing="ij")
    pcm = axes[1].pcolormesh(X, T, U, shading="auto", cmap="inferno")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title(r"Champ $u(x,t)$ — chaleur")
    fig.colorbar(pcm, ax=axes[1], label="u")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig01_chaleur_evolution.png")
    plt.close(fig)


def fig02_wave_string() -> None:
    """Corde vibrante u(x,t) — leapfrog."""
    c = 1.0
    x, t, U = wave_leapfrog(c, nx=220, t_final=2.0, cfl=0.9)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # demi-période T = 2L/c = 2
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ti in times:
        idx = int(np.argmin(np.abs(t - ti)))
        axes[0].plot(x, U[idx], label=f"t = {t[idx]:.2f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x,t)")
    axes[0].set_title("Corde vibrante — snapshots")
    axes[0].legend(fontsize=9)

    # sous-échantillonner pour lisibilité
    step_t = max(1, len(t) // 200)
    step_x = max(1, len(x) // 150)
    Ts = t[::step_t]
    Xs = x[::step_x]
    Us = U[::step_t, ::step_x]
    Tm, Xm = np.meshgrid(Ts, Xs, indexing="ij")
    pcm = axes[1].pcolormesh(Xm, Tm, Us, shading="auto", cmap="RdBu_r")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title(r"Champ $u(x,t)$ — ondes")
    fig.colorbar(pcm, ax=axes[1], label="u")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig02_onde_corde.png")
    plt.close(fig)


def fig03_erreur_l2_ordre() -> None:
    """Erreur L2 vs Δx — pente = ordre."""
    kappa = 0.1
    t_final = 0.1
    nxs = np.array([20, 40, 80, 160, 320])
    errs_cn = []
    errs_ftcs = []
    dxs = []
    for nx in nxs:
        dx = 1.0 / nx
        dxs.append(dx)
        # nt assez grand pour que l'erreur temporelle soit négligeable
        nt = max(200, int(4 * kappa * t_final / dx**2))
        # garantir r ≤ 0.4 pour FTCS
        r_target = 0.4
        nt_ftcs = max(nt, int(np.ceil(kappa * t_final / (r_target * dx**2))))
        x, t, U = heat_crank_nicolson(kappa, nx=int(nx), t_final=t_final, nt=nt_ftcs)
        errs_cn.append(l2_error(U[-1], heat_exact_sine(x, t[-1], kappa), dx))
        x, t, U = heat_ftcs(kappa, nx=int(nx), t_final=t_final, nt=nt_ftcs)
        errs_ftcs.append(l2_error(U[-1], heat_exact_sine(x, t[-1], kappa), dx))

    dxs = np.array(dxs)
    errs_cn = np.array(errs_cn)
    errs_ftcs = np.array(errs_ftcs)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.loglog(dxs, errs_cn, "o-", label="Crank–Nicolson", color="#1f4e79")
    ax.loglog(dxs, errs_ftcs, "s-", label="FTCS", color="#c45c26")
    # droites de référence
    ref = errs_cn[0] * (dxs / dxs[0]) ** 2
    ax.loglog(dxs, ref, "k--", alpha=0.7, label=r"réf. $O(\Delta x^2)$")
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"erreur $L^2$ à $t=0.1$")
    ax.set_title("Convergence spatiale — erreur L2 vs Δx")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig03_erreur_l2_vs_dx.png")
    plt.close(fig)

    # sauvegarder ordres pour annexe
    order_cn = np.log(errs_cn[-2] / errs_cn[-1]) / np.log(2)
    order_ft = np.log(errs_ftcs[-2] / errs_ftcs[-1]) / np.log(2)
    (FIGDIR / "orders.txt").write_text(
        f"order_cn={order_cn:.4f}\norder_ftcs={order_ft:.4f}\n"
        f"errs_cn={errs_cn.tolist()}\nerrs_ftcs={errs_ftcs.tolist()}\n",
        encoding="utf-8",
    )


def fig04_domaine_stabilite() -> None:
    """Domaine de stabilité von Neumann pour FTCS (r ≤ 1/2)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    # axe r = κ dt / dx²
    r_vals = np.linspace(0, 1.2, 400)
    # facteur d'amplification max |g| = |1 - 4 r sin²(θ/2)| avec max en sin=1
    g_max = np.abs(1 - 4 * r_vals)
    ax.plot(r_vals, g_max, color="#1f4e79", lw=2, label=r"$|g|_{\max}=|1-4r|$")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="seuil |g|=1")
    ax.axvline(0.5, color="#c45c26", ls=":", lw=1.8, label=r"$r=1/2$")
    ax.fill_between(
        r_vals,
        0,
        1.0,
        where=(r_vals <= 0.5),
        color="#2a9d8f",
        alpha=0.25,
        label="zone stable",
    )
    ax.fill_between(
        r_vals,
        1.0,
        g_max,
        where=(r_vals > 0.5),
        color="#e76f51",
        alpha=0.35,
        label="zone instable",
    )
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 4.0)
    ax.set_xlabel(r"nombre de Fourier $r=\kappa\Delta t/\Delta x^2$")
    ax.set_ylabel(r"$|g|_{\max}$ (mode le plus haut)")
    ax.set_title("Domaine de stabilité — schéma FTCS explicite")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig04_domaine_stabilite_ftcs.png")
    plt.close(fig)


def fig05_comparaison_schemas() -> None:
    """Comparaison FTCS / BTCS / CN : diffusion numérique."""
    kappa = 0.05
    nx = 80
    t_final = 0.3
    # r ≈ 0.45 pour FTCS
    dx = 1.0 / nx
    r = 0.45
    dt = r * dx**2 / kappa
    nt = int(np.round(t_final / dt))
    dt = t_final / nt

    solvers = {
        "FTCS": heat_ftcs,
        "BTCS": heat_btcs,
        "CN": heat_crank_nicolson,
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x_ref = np.linspace(0, 1, nx + 1)
    u_ex = heat_exact_sine(x_ref, t_final, kappa)
    axes[0].plot(x_ref, u_ex, "k-", lw=2, label="exact")

    errs = {}
    for name, solver in solvers.items():
        x, t, U = solver(kappa, nx=nx, t_final=t_final, nt=nt)
        axes[0].plot(x, U[-1], "--", label=name)
        errs[name] = l2_error(U[-1], heat_exact_sine(x, t[-1], kappa), x[1] - x[0])

    axes[0].set_xlabel("x")
    axes[0].set_ylabel(f"u(x, t={t_final})")
    axes[0].set_title("Profil final — comparaison des schémas")
    axes[0].legend(fontsize=9)

    names = list(errs.keys())
    vals = [errs[n] for n in names]
    colors = ["#c45c26", "#2a9d8f", "#1f4e79"]
    axes[1].bar(names, vals, color=colors)
    axes[1].set_ylabel(r"erreur $L^2$")
    axes[1].set_title(f"Erreurs à t={t_final} (r={r:.2f})")
    for i, v in enumerate(vals):
        axes[1].text(i, v * 1.02, f"{v:.2e}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig05_comparaison_schemas.png")
    plt.close(fig)


def fig06_schema_conceptuel() -> None:
    """Bloc-diagramme conceptuel du modèle."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.4, 4.2, 2.4, 1.2, "Physique\nbarre / corde"),
        (3.5, 4.2, 2.8, 1.2, "EDP continue\nchaleur / ondes"),
        (7.0, 4.2, 2.6, 1.2, "CI + CL\nDirichlet"),
        (0.4, 1.6, 2.6, 1.4, "Discrétisation\nΔx, Δt"),
        (3.5, 1.6, 2.8, 1.4, "Schéma DF\nFTCS/BTCS/CN\nou leapfrog"),
        (7.0, 1.6, 2.6, 1.4, "Analyse\nstabilité,\nordre, erreur"),
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#e8f1f8",
            edgecolor="#1f4e79",
            lw=1.5,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((2.8, 4.8), (3.5, 4.8)),
        ((6.3, 4.8), (7.0, 4.8)),
        ((4.9, 4.2), (4.9, 3.0)),
        ((2.8, 2.3), (3.5, 2.3)),
        ((6.3, 2.3), (7.0, 2.3)),
        ((1.7, 4.2), (1.7, 3.0)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#333333",
                lw=1.3,
            )
        )
    ax.set_title(
        "Schéma conceptuel : de la physique à l'analyse numérique",
        fontsize=12,
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig06_schema_conceptuel.png")
    plt.close(fig)


def fig07_ftcs_instabilite() -> None:
    """Instabilité FTCS lorsque r > 1/2."""
    kappa = 0.1
    nx = 40
    t_final = 0.05
    dx = 1.0 / nx

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, r, title in (
        (axes[0], 0.4, r"FTCS stable ($r=0.4$)"),
        (axes[1], 0.6, r"FTCS instable ($r=0.6$)"),
    ):
        dt = r * dx**2 / kappa
        nt = max(1, int(np.round(t_final / dt)))
        x, t, U = heat_ftcs(kappa, nx=nx, t_final=t_final, nt=nt)
        # limiter l'affichage si explosion
        U_plot = np.clip(U, -5, 5)
        T, X = np.meshgrid(t, x, indexing="ij")
        pcm = ax.pcolormesh(X, T, U_plot, shading="auto", cmap="coolwarm")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, label="u (clip ±5)")
    fig.suptitle("Effet du nombre de Fourier sur FTCS", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig07_ftcs_instabilite.png", bbox_inches="tight")
    plt.close(fig)


def fig08_cfl_ondes() -> None:
    """Influence du CFL sur l'onde leapfrog."""
    c = 1.0
    nx = 160
    t_final = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for cfl, color, ls in ((0.9, "#1f4e79", "-"), (1.0, "#2a9d8f", "--")):
        x, t, U = wave_leapfrog(c, nx=nx, t_final=t_final, cfl=cfl)
        axes[0].plot(x, U[-1], ls, color=color, label=f"CFL={cfl}")
    x_ex = np.linspace(0, 1, nx + 1)
    axes[0].plot(x_ex, wave_exact_sine(x_ex, t_final, c), "k:", lw=2, label="exact")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(f"u(x, t={t_final})")
    axes[0].set_title("Leapfrog sous CFL ≤ 1")
    axes[0].legend(fontsize=9)

    # CFL > 1 : explosion
    x, t, U = wave_leapfrog(c, nx=80, t_final=0.6, cfl=1.15)
    # norme L2 en temps
    dx = x[1] - x[0]
    norms = np.sqrt(dx * np.sum(U**2, axis=1))
    axes[1].semilogy(t, np.maximum(norms, 1e-16), color="#c45c26")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\|u(\cdot,t)\|_{L^2}$")
    axes[1].set_title("CFL = 1.15 : croissance explosive")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig08_cfl_ondes.png")
    plt.close(fig)


def fig09_dispersion_numerique() -> None:
    """Dispersion / diffusion numérique : paquet d'ondes vs mode propre."""
    # Chaleur : amortissement exact vs numérique pour mode k=π
    kappa = 0.1
    L = 1.0
    nx = 60
    t_final = 0.25
    dx = L / nx
    r = 0.4
    dt = r * dx**2 / kappa
    nt = int(np.round(t_final / dt))
    t_final = nt * dt

    x, t, U_ft = heat_ftcs(kappa, nx=nx, t_final=t_final, nt=nt)
    _, _, U_bt = heat_btcs(kappa, nx=nx, t_final=t_final, nt=nt)
    _, _, U_cn = heat_crank_nicolson(kappa, nx=nx, t_final=t_final, nt=nt)

    def amp(U):
        # amplitude du mode sin(πx) via produit scalaire
        mode = np.sin(np.pi * x / L)
        return (U @ mode) * dx * 2  # normalisation ≈1 à t=0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    a_ex = np.exp(-kappa * (np.pi / L) ** 2 * t)
    ax.plot(t, a_ex, "k-", lw=2, label="exact")
    ax.plot(t, amp(U_ft), label="FTCS")
    ax.plot(t, amp(U_bt), label="BTCS")
    ax.plot(t, amp(U_cn), label="CN")
    ax.set_xlabel("t")
    ax.set_ylabel("amplitude du mode sin(πx)")
    ax.set_title("Diffusion numérique : amortissement modal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig09_dispersion_diffusion.png")
    plt.close(fig)


def fig10_energie_onde() -> None:
    """Énergie discrète de la corde (quasi-conservation leapfrog)."""
    c = 1.0
    x, t, U = wave_leapfrog(c, nx=200, t_final=2.0, cfl=0.95)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    # énergie cinétique approx + potentielle
    # E ≈ 1/2 ∫ (u_t² + c² u_x²) dx
    ut = np.zeros_like(U)
    ut[1:-1] = (U[2:] - U[:-2]) / (2 * dt)
    ut[0] = (U[1] - U[0]) / dt
    ut[-1] = (U[-1] - U[-2]) / dt
    ux = np.zeros_like(U)
    ux[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2 * dx)
    energy = 0.5 * dx * np.sum(ut**2 + (c**2) * ux**2, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t, energy, color="#1f4e79")
    ax.set_xlabel("t")
    ax.set_ylabel("énergie discrète E(t)")
    ax.set_title("Quasi-conservation d'énergie — leapfrog (CFL=0.95)")
    ax.set_ylim(0, 1.2 * np.max(energy))
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig10_energie_onde.png")
    plt.close(fig)


def main() -> None:
    fig01_heat_evolution()
    fig02_wave_string()
    fig03_erreur_l2_ordre()
    fig04_domaine_stabilite()
    fig05_comparaison_schemas()
    fig06_schema_conceptuel()
    fig07_ftcs_instabilite()
    fig08_cfl_ondes()
    fig09_dispersion_numerique()
    fig10_energie_onde()
    pngs = sorted(FIGDIR.glob("fig*.png"))
    print(f"Generated {len(pngs)} figures in {FIGDIR}")
    for p in pngs:
        print(" ", p.name)


if __name__ == "__main__":
    main()
