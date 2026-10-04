#!/usr/bin/env python3
"""Génère les figures du rapport 03-vol-smile (PNG)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from implied_vol import bs_call, implied_vol_call, implied_vol_newton
from svi import (
    butterfly_arbitrage_free,
    build_svi_surface,
    calibrate_svi,
    make_synthetic_smile,
    svi_density_proxy,
    svi_total_variance,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Paramètres de référence (Annexe B du rapport)
S0, R, T_REF = 100.0, 0.01, 0.5
F_REF = S0 * np.exp(R * T_REF)
TRUE_PARAMS = (0.03, 0.25, -0.35, 0.05, 0.25)
NOISE = 1e-4
SEED = 0


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fig01_smile_iv() -> None:
    K, iv, _ = make_synthetic_smile(
        F=F_REF, T=T_REF, true_params=TRUE_PARAMS, n_strikes=41, noise=0.0
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(K, 100 * iv, "o-", color="#0b3d91", lw=2, ms=4, label=r"$\sigma_{\mathrm{imp}}(K)$")
    ax.axvline(F_REF, color="#c0392b", ls="--", lw=1.2, label=f"Forward F={F_REF:.2f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel(r"Volatilité implicite (%)")
    ax.set_title(r"Smile de volatilité implicite — maturité $T=0.5$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_smile_iv.png")
    plt.close(fig)


def fig02_svi_fit() -> None:
    K, iv_mkt, w_mkt = make_synthetic_smile(
        F=F_REF, T=T_REF, true_params=TRUE_PARAMS, n_strikes=21, noise=NOISE, seed=SEED
    )
    k = np.log(K / F_REF)
    fit = calibrate_svi(k, w_mkt)
    k_fine = np.linspace(k.min() - 0.05, k.max() + 0.05, 200)
    w_fit = svi_total_variance(
        k_fine, fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"]
    )
    iv_fit = np.sqrt(np.maximum(w_fit, 1e-12) / T_REF)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(K, 100 * iv_mkt, c="#1a5276", s=36, zorder=3, label="Marché synthétique")
    ax.plot(
        F_REF * np.exp(k_fine),
        100 * iv_fit,
        color="#e67e22",
        lw=2.2,
        label=f"Fit SVI (RMSE$_w$={fit['rmse']:.2e})",
    )
    ax.set_xlabel("Strike K")
    ax.set_ylabel(r"$\sigma_{\mathrm{imp}}$ (%)")
    ax.set_title("Calibration SVI vs smile de marché (synthétique)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_svi_fit.png")
    plt.close(fig)


def fig03_residuals() -> None:
    K, _, w_mkt = make_synthetic_smile(
        F=F_REF, T=T_REF, true_params=TRUE_PARAMS, n_strikes=21, noise=NOISE, seed=SEED
    )
    k = np.log(K / F_REF)
    fit = calibrate_svi(k, w_mkt)
    w_fit = svi_total_variance(k, fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"])
    resid = w_fit - w_mkt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.stem(k, resid, linefmt="C0-", markerfmt="C0o", basefmt=" ")
    ax.set_xlabel(r"Log-moneyness $k=\log(K/F)$")
    ax.set_ylabel(r"Résidu $w_{\mathrm{SVI}}-w_{\mathrm{mkt}}$")
    ax.set_title("Résidus de calibration SVI (total variance)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_residuals.png")
    plt.close(fig)


def fig04_surface() -> None:
    maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    # smile qui s'aplatit légèrement avec T (réaliste)
    params = [
        (0.02, 0.28, -0.45, 0.02, 0.22),
        (0.03, 0.25, -0.35, 0.05, 0.25),
        (0.05, 0.22, -0.30, 0.04, 0.28),
        (0.07, 0.20, -0.25, 0.03, 0.30),
        (0.09, 0.18, -0.20, 0.02, 0.32),
    ]
    k_grid, Ts, W = build_svi_surface(maturities, params, F=F_REF)
    KK, TT = np.meshgrid(F_REF * np.exp(k_grid), Ts)
    Sigma = np.sqrt(np.maximum(W, 1e-12) / TT)
    fig = plt.figure(figsize=(8.0, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(KK, TT, 100 * Sigma, cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturité T")
    ax.set_zlabel(r"$\sigma$ (%)")
    ax.set_title(r"Surface de volatilité implicite $\sigma(K,T)$")
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label="%")
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_surface_sigma.png")
    plt.close(fig)


def fig05_sensitivity() -> None:
    k = np.linspace(-0.5, 0.5, 200)
    a, b, rho, m, sig = TRUE_PARAMS
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), sharex=True)
    # rho
    for rho_i in (-0.8, -0.35, 0.0, 0.5):
        w = svi_total_variance(k, a, b, rho_i, m, sig)
        axes[0, 0].plot(k, w, label=rf"$\rho={rho_i}$")
    axes[0, 0].set_title("Sensibilité à ρ (skew)")
    axes[0, 0].set_ylabel("w(k)")
    axes[0, 0].legend(fontsize=8)
    # sigma
    for s_i in (0.1, 0.25, 0.5, 0.9):
        w = svi_total_variance(k, a, b, rho, m, s_i)
        axes[0, 1].plot(k, w, label=rf"$\sigma={s_i}$")
    axes[0, 1].set_title("Sensibilité à σ (courbure ATM)")
    axes[0, 1].legend(fontsize=8)
    # b
    for b_i in (0.1, 0.25, 0.4, 0.7):
        w = svi_total_variance(k, a, b_i, rho, m, sig)
        axes[1, 0].plot(k, w, label=rf"$b={b_i}$")
    axes[1, 0].set_title("Sensibilité à b (pentes asymptotiques)")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("w(k)")
    axes[1, 0].legend(fontsize=8)
    # m
    for m_i in (-0.2, 0.0, 0.05, 0.25):
        w = svi_total_variance(k, a, b, rho, m_i, sig)
        axes[1, 1].plot(k, w, label=rf"$m={m_i}$")
    axes[1, 1].set_title("Sensibilité à m (translation)")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Sensibilité des paramètres SVI raw", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_svi_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def fig06_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, 1.3, "Prix marché\nC(K,T)"),
        (2.5, 1.3, "Inversion BS\n(Brent)"),
        (4.7, 1.3, r"Smile" "\n" r"σ_imp(K)"),
        (6.9, 1.3, "Calibration\nSVI"),
        (2.5, 0.2, r"Surface w(k,T)"),
        (6.9, 0.2, "Pricing /\nrisques"),
    ]
    for x, y, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            1.8,
            1.1,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#d6eaf8",
            edgecolor="#1a5276",
            lw=1.5,
        )
        ax.add_patch(patch)
        ax.text(x + 0.9, y + 0.55, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((2.1, 1.85), (2.5, 1.85)),
        ((4.3, 1.85), (4.7, 1.85)),
        ((6.5, 1.85), (6.9, 1.85)),
        ((5.6, 1.3), (4.3, 1.0)),
        ((7.8, 1.3), (7.8, 1.3)),
    ]
    for (x1, y1), (x2, y2) in [
        ((2.1, 1.85), (2.5, 1.85)),
        ((4.3, 1.85), (4.7, 1.85)),
        ((6.5, 1.85), (6.9, 1.85)),
        ((5.6, 1.3), (3.4, 1.15)),
        ((7.8, 1.3), (7.8, 1.3)),
    ][:4]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#2c3e50",
                lw=1.4,
            )
        )
    ax.annotate(
        "",
        xy=(3.4, 1.3),
        xytext=(5.6, 1.3),
        arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.4),
    )
    # flèche smile → surface
    ax.annotate(
        "",
        xy=(3.4, 1.3),
        xytext=(5.6, 1.5),
        arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.2),
    )
    ax.annotate(
        "",
        xy=(3.4, 0.75),
        xytext=(5.6, 1.3),
        arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.2),
    )
    ax.annotate(
        "",
        xy=(6.9, 0.75),
        xytext=(7.8, 1.3),
        arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.2),
    )
    ax.set_title("Pipeline conceptuel : prix → smile → SVI → surface → pricing", pad=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_pipeline.png")
    plt.close(fig)


def fig07_iv_methods() -> None:
    strikes = np.linspace(70, 130, 25)
    true_sig = 0.22
    errs_brent, errs_newton = [], []
    for K in strikes:
        price = bs_call(S0, float(K), T_REF, R, true_sig)
        iv_b = implied_vol_call(price, S0, float(K), T_REF, R)
        iv_n = implied_vol_newton(price, S0, float(K), T_REF, R, sigma0=0.2)
        errs_brent.append(abs(iv_b - true_sig))
        errs_newton.append(abs(iv_n - true_sig))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(strikes, errs_brent, "o-", label="Brent", color="#1a5276")
    ax.semilogy(strikes, errs_newton, "s--", label="Newton–Raphson", color="#e67e22")
    ax.set_xlabel("Strike K")
    ax.set_ylabel(r"$|\sigma_{\mathrm{rec}}-\sigma_{\mathrm{true}}|$")
    ax.set_title("Comparaison Brent vs Newton (round-trip BS, σ=22%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_iv_brent_vs_newton.png")
    plt.close(fig)


def fig08_butterfly() -> None:
    k = np.linspace(-0.6, 0.6, 301)
    a, b, rho, m, sig = TRUE_PARAMS
    g_ok = svi_density_proxy(k, a, b, rho, m, sig)
    # paramètres agressifs pouvant créer butterfly (b trop grand + |rho|≈1)
    bad = (0.01, 1.2, -0.99, 0.0, 0.05)
    g_bad = svi_density_proxy(k, *bad)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(k, g_ok, color="#196f3d", lw=2, label="SVI sain (g≥0)")
    ax.plot(k, g_bad, color="#922b21", lw=2, label="SVI agressif (g<0 possible)")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xlabel("k")
    ax.set_ylabel("g(k) (proxy densité / butterfly)")
    ax.set_title("Condition d'absence d'arbitrage butterfly (g(k)≥0)")
    ax.legend()
    ok = butterfly_arbitrage_free(k, a, b, rho, m, sig)
    bad_ok = butterfly_arbitrage_free(k, *bad)
    ax.text(
        0.02,
        0.05,
        f"sain free={ok}, agressif free={bad_ok}",
        transform=ax.transAxes,
        fontsize=9,
        color="#2c3e50",
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "08_butterfly_g.png")
    plt.close(fig)


def fig09_calendar() -> None:
    k = np.linspace(-0.4, 0.4, 120)
    params_short = (0.04, 0.30, -0.4, 0.0, 0.2)
    params_long = (0.08, 0.22, -0.3, 0.0, 0.25)
    # cas calendaire OK
    w1 = svi_total_variance(k, *params_short)
    w2 = svi_total_variance(k, *params_long)
    # cas calendaire cassé : w court > w long sur une zone
    w2_bad = svi_total_variance(k, 0.02, 0.15, -0.2, 0.0, 0.3)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    axes[0].plot(k, w1, label="T=0.5", color="#1a5276")
    axes[0].plot(k, w2, label="T=1.0", color="#e67e22")
    axes[0].set_title("Calendaire OK : w croît avec T")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("w(k,T)")
    axes[0].legend()
    axes[1].plot(k, w1, label="T=0.5", color="#1a5276")
    axes[1].plot(k, w2_bad, label="T=1.0 (mauvais)", color="#922b21")
    axes[1].fill_between(
        k,
        w2_bad,
        w1,
        where=(w1 > w2_bad),
        color="#f5b7b1",
        alpha=0.6,
        label="zone d'arbitrage",
    )
    axes[1].set_title("Calendaire violé : w(T1)>w(T2)")
    axes[1].set_xlabel("k")
    axes[1].legend(fontsize=8)
    fig.suptitle("Arbitrage calendaire et total variance", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "09_calendar_arbitrage.png", bbox_inches="tight")
    plt.close(fig)


def fig10_total_variance() -> None:
    K, iv, w = make_synthetic_smile(
        F=F_REF, T=T_REF, true_params=TRUE_PARAMS, n_strikes=41, noise=0.0
    )
    k = np.log(K / F_REF)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(k, w, "o-", color="#0b3d91", lw=2, ms=4)
    ax.set_xlabel(r"$k=\log(K/F)$")
    ax.set_ylabel(r"Total variance $w=\sigma_{\mathrm{imp}}^2 T$")
    ax.set_title("Total variance SVI (objet calibré)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "10_total_variance.png")
    plt.close(fig)


def main() -> None:
    _style()
    fig01_smile_iv()
    fig02_svi_fit()
    fig03_residuals()
    fig04_surface()
    fig05_sensitivity()
    fig06_pipeline()
    fig07_iv_methods()
    fig08_butterfly()
    fig09_calendar()
    fig10_total_variance()
    print(f"Figures écrites dans {FIGDIR}")
    for p in sorted(FIGDIR.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
