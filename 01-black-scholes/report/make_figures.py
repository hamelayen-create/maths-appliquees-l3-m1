#!/usr/bin/env python3
"""Génère les figures du rapport Black–Scholes (backend Agg)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bs_closed_form import call_price, delta_call, put_price
from bs_mc import price_call_mc
from bs_pde import price_call_crank_nicolson

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Paramètres de référence (Annexe B)
S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
SEED = 42


def save(fig: plt.Figure, name: str) -> None:
    path = FIGDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def fig01_gbm_paths() -> None:
    rng = np.random.default_rng(SEED)
    n_paths, n_steps = 12, 252
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for i in range(n_steps):
        z = rng.standard_normal(n_paths)
        paths[:, i + 1] = paths[:, i] * np.exp(
            (R - 0.5 * SIGMA**2) * dt + SIGMA * np.sqrt(dt) * z
        )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(n_paths):
        ax.plot(t, paths[i], lw=1.0, alpha=0.75)
    ax.axhline(K, color="k", ls="--", lw=1.2, label=f"Strike K={K:.0f}")
    ax.set_xlabel("Temps t (années)")
    ax.set_ylabel("Spot $S_t$")
    ax.set_title("Trajectoires de mouvement brownien géométrique (GBM)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    save(fig, "01_gbm_trajectories.png")


def fig02_call_vs_K() -> None:
    strikes = np.linspace(60, 140, 81)
    prices = [call_price(S0, k, T, R, SIGMA) for k in strikes]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(strikes, prices, color="#1a5276", lw=2.2, label="Call BS")
    ax.axvline(S0, color="gray", ls=":", label=f"$S_0={S0:.0f}$")
    ax.set_xlabel("Strike $K$")
    ax.set_ylabel("Prix du call $C(S_0,K)$")
    ax.set_title("Prix du call européen en fonction du strike")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "02_call_vs_K.png")


def fig03_call_vs_sigma() -> None:
    sigmas = np.linspace(0.05, 0.80, 76)
    prices = [call_price(S0, K, T, R, s) for s in sigmas]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(sigmas, prices, color="#117a65", lw=2.2)
    ax.axvline(SIGMA, color="gray", ls=":", label=rf"$\sigma={SIGMA}$")
    ax.set_xlabel(r"Volatilité $\sigma$")
    ax.set_ylabel("Prix du call")
    ax.set_title(r"Prix du call ATM en fonction de $\sigma$ (effet Vega)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "03_call_vs_sigma.png")


def fig04_mc_convergence() -> None:
    closed = call_price(S0, K, T, R, SIGMA)
    n_list = np.unique(np.logspace(2.5, 5.5, 18).astype(int))
    # Forcer parité pour antithetic
    n_list = np.where(n_list % 2 == 0, n_list, n_list + 1)
    est, lo, hi = [], [], []
    for n in n_list:
        m, se = price_call_mc(S0, K, T, R, SIGMA, n_paths=int(n), seed=SEED)
        est.append(m)
        lo.append(m - 1.96 * se)
        hi.append(m + 1.96 * se)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(n_list, lo, hi, color="#5dade2", alpha=0.35, label="IC 95 %")
    ax.plot(n_list, est, "o-", color="#1a5276", ms=4, label="Estimateur MC")
    ax.axhline(closed, color="#c0392b", ls="--", lw=1.5, label=f"BS fermé = {closed:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Nombre de trajectoires $N$")
    ax.set_ylabel("Prix call estimé")
    ax.set_title("Convergence Monte Carlo (antithetic) ± intervalle de confiance")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "04_mc_convergence.png")


def fig05_pde_errors() -> None:
    closed = call_price(S0, K, T, R, SIGMA)
    n_space_list = [40, 60, 80, 120, 160, 200, 280, 360]
    err_space = []
    for ns in n_space_list:
        pde = price_call_crank_nicolson(
            S0, K, T, R, SIGMA, n_space=ns, n_time=300
        )
        err_space.append(abs(pde - closed))

    n_time_list = [20, 40, 60, 100, 150, 200, 300, 400]
    err_time = []
    for nt in n_time_list:
        pde = price_call_crank_nicolson(
            S0, K, T, R, SIGMA, n_space=300, n_time=nt
        )
        err_time.append(abs(pde - closed))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].loglog(n_space_list, err_space, "s-", color="#1a5276")
    axes[0].set_xlabel(r"$n_{\mathrm{space}}$")
    axes[0].set_ylabel(r"$|V_{\mathrm{PDE}}-V_{\mathrm{BS}}|$")
    axes[0].set_title("Erreur EDP vs pas d'espace")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].loglog(n_time_list, err_time, "o-", color="#117a65")
    axes[1].set_xlabel(r"$n_{\mathrm{time}}$")
    axes[1].set_ylabel(r"$|V_{\mathrm{PDE}}-V_{\mathrm{BS}}|$")
    axes[1].set_title("Erreur EDP vs pas de temps")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.suptitle("Convergence du schéma Crank–Nicolson (log-spot)", y=1.02)
    fig.tight_layout()
    save(fig, "05_pde_errors.png")


def fig06_price_surface() -> None:
    S_grid = np.linspace(40, 160, 80)
    t_grid = np.linspace(0.02, T, 50)
    SS, TT = np.meshgrid(S_grid, t_grid)
    VV = np.zeros_like(SS)
    for i in range(TT.shape[0]):
        for j in range(TT.shape[1]):
            tau = T - TT[i, j]  # TT = calendar time from 0 to T; tau = maturity left
            # On veut V(S,t) avec t courant : maturité restante = T-t
            rem = max(T - TT[i, j], 1e-6)
            VV[i, j] = call_price(float(SS[i, j]), K, rem, R, SIGMA)

    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        SS, TT, VV, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95
    )
    ax.set_xlabel("Spot $S$")
    ax.set_ylabel("Temps $t$")
    ax.set_zlabel("Prix $V(S,t)$")
    ax.set_title("Surface de prix du call Black–Scholes $V(S,t)$")
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="Prix")
    save(fig, "06_price_surface.png")


def fig07_model_diagram() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.4, 3.5, 2.2, 1.4, "Hypothèses\nGBM, r, σ\nmarché complet"),
        (3.2, 3.5, 2.2, 1.4, "Portefeuille\nrépliquant\nΔ·S + B"),
        (6.0, 3.5, 2.2, 1.4, "EDP BS\n∂tV + … = 0\nV(T)=(S−K)+"),
        (3.2, 0.6, 2.2, 1.4, "Formule\nfermée\nC = SN(d1)−…"),
        (6.0, 0.6, 2.2, 1.4, "Numérique\nCN / MC"),
    ]
    for x, y, w, h, txt in boxes:
        rect = plt.Rectangle(
            (x, y), w, h, fill=True, facecolor="#d6eaf8", edgecolor="#1a5276", lw=1.8
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#1a5276", lw=1.6),
        )

    arrow(2.6, 4.2, 3.2, 4.2)
    arrow(5.4, 4.2, 6.0, 4.2)
    arrow(7.1, 3.5, 7.1, 2.0)
    arrow(4.3, 3.5, 4.3, 2.0)
    ax.text(5.0, 5.5, "Fil conducteur du modèle Black–Scholes", ha="center", fontsize=12)
    save(fig, "07_model_diagram.png")


def fig08_greeks() -> None:
    S = np.linspace(50, 150, 200)
    d1 = (np.log(S / K) + (R + 0.5 * SIGMA**2) * T) / (SIGMA * np.sqrt(T))
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * SIGMA * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0  # par point de vol (1%)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].plot(S, delta, color="#1a5276", lw=2)
    axes[0].set_title(r"Delta $\Delta=\partial_S C$")
    axes[0].set_xlabel("$S$")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(S, gamma, color="#117a65", lw=2)
    axes[1].set_title(r"Gamma $\Gamma=\partial_{SS} C$")
    axes[1].set_xlabel("$S$")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(S, vega, color="#af601a", lw=2)
    axes[2].set_title(r"Vega (par 1% de $\sigma$)")
    axes[2].set_xlabel("$S$")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Grecs du call européen Black–Scholes (ATM params)", y=1.05)
    fig.tight_layout()
    save(fig, "08_greeks.png")


def fig09_put_call_parity() -> None:
    S = np.linspace(60, 140, 100)
    calls = np.array([call_price(s, K, T, R, SIGMA) for s in S])
    puts = np.array([put_price(s, K, T, R, SIGMA) for s in S])
    forward = S - K * np.exp(-R * T)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(S, calls - puts, lw=2.2, label="$C-P$")
    ax.plot(S, forward, "--", lw=1.8, label=r"$S-Ke^{-rT}$")
    ax.set_xlabel("Spot $S$")
    ax.set_ylabel("Valeur")
    ax.set_title("Vérification numérique de la parité call–put")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "09_put_call_parity.png")


def fig10_methods_comparison() -> None:
    closed = call_price(S0, K, T, R, SIGMA)
    pde = price_call_crank_nicolson(S0, K, T, R, SIGMA, n_space=250, n_time=250)
    mc, se = price_call_mc(S0, K, T, R, SIGMA, n_paths=200_000, seed=SEED)
    methods = ["Formule\nfermée", "EDP\nCrank–Nicolson", "Monte Carlo\n(antithetic)"]
    values = [closed, pde, mc]
    errors = [0.0, abs(pde - closed), abs(mc - closed)]
    colors = ["#1a5276", "#117a65", "#af601a"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(methods, values, color=colors, edgecolor="k", lw=0.6)
    axes[0].set_ylabel("Prix du call ATM")
    axes[0].set_title("Comparaison des trois estimateurs")
    axes[0].axhline(closed, color="gray", ls=":", lw=1)
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.05, f"{v:.4f}", ha="center", fontsize=9)

    axes[1].bar(methods, errors, color=colors, edgecolor="k", lw=0.6)
    axes[1].set_ylabel(r"$|$erreur vs BS$|$")
    axes[1].set_title("Écart absolu à la formule fermée")
    axes[1].text(
        2,
        errors[2] + 0.002,
        f"IC95 ±{1.96*se:.4f}",
        ha="center",
        fontsize=8,
        color="#af601a",
    )
    fig.tight_layout()
    save(fig, "10_methods_comparison.png")


def main() -> None:
    print(f"Generating figures in {FIGDIR}")
    fig01_gbm_paths()
    fig02_call_vs_K()
    fig03_call_vs_sigma()
    fig04_mc_convergence()
    fig05_pde_errors()
    fig06_price_surface()
    fig07_model_diagram()
    fig08_greeks()
    fig09_put_call_parity()
    fig10_methods_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
