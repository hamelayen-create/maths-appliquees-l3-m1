#!/usr/bin/env python3
"""Génère les figures du rapport Markowitz (PNG dans report/figures/)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio import (  # noqa: E402
    backtest_equity,
    efficient_frontier_analytic,
    efficient_frontier_long_only,
    gmv_weights,
    ledoit_wolf_cov,
    max_sharpe_weights,
    optimize_long_only,
    portfolio_stats,
    sample_covariance,
    sharpe_ratio,
    simulate_returns,
)

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Style sobre, lisible (polycopié)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 140,
    }
)

SEED = 7
N_ASSETS = 5
LABELS = [f"A{i+1}" for i in range(N_ASSETS)]


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig01_risk_return_cloud() -> None:
    R, mu, cov = simulate_returns(n_assets=N_ASSETS, seed=SEED)
    vols = np.sqrt(np.diag(cov))
    # nuage de portefeuilles aléatoires long-only (Dirichlet)
    rng = np.random.default_rng(SEED)
    W = rng.dirichlet(np.ones(N_ASSETS), size=800)
    rets = W @ mu
    risks = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(risks, rets, s=8, c="#7a8b99", alpha=0.45, label="Portefeuilles aléatoires (long-only)")
    ax.scatter(vols, mu, s=90, c="#c0392b", zorder=5, label="Actifs individuels")
    for i, lab in enumerate(LABELS):
        ax.annotate(lab, (vols[i], mu[i]), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel(r"Risque $\sigma_p$ (vol. annualisée)")
    ax.set_ylabel(r"Rendement $\mu_p$ (annualisé)")
    ax.set_title("Nuage risque–rendement des actifs et portefeuilles")
    ax.legend(loc="best")
    _save(fig, "01_nuage_risque_rendement.png")


def fig02_frontier_analytic_vs_long_only() -> None:
    _, mu, cov = simulate_returns(n_assets=N_ASSETS, seed=SEED)
    t_a, v_a, _ = efficient_frontier_analytic(mu, cov, n_points=60)
    t_l, v_l, _ = efficient_frontier_long_only(mu, cov, n_points=30)
    vols = np.sqrt(np.diag(cov))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(v_a, t_a, color="#1f4e79", lw=2.0, label="Frontière analytique (short ok)")
    ax.plot(v_l, t_l, color="#e67e22", lw=2.2, label="Frontière long-only (QP)")
    ax.scatter(vols, mu, s=70, c="#c0392b", zorder=5, label="Actifs")
    for i, lab in enumerate(LABELS):
        ax.annotate(lab, (vols[i], mu[i]), textcoords="offset points", xytext=(5, 3))
    ax.set_xlabel(r"Risque $\sigma_p$")
    ax.set_ylabel(r"Rendement $\mu_p$")
    ax.set_title("Frontière efficiente : analytique vs long-only")
    ax.legend(loc="best")
    _save(fig, "02_frontiere_analytique_vs_long_only.png")


def fig03_weights_along_frontier() -> None:
    _, mu, cov = simulate_returns(n_assets=N_ASSETS, seed=SEED)
    targets, _, W = efficient_frontier_long_only(mu, cov, n_points=28)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.stackplot(targets, W.T, labels=LABELS, alpha=0.9)
    ax.set_xlabel(r"Rendement cible $\mu_p$")
    ax.set_ylabel("Poids $w_i$")
    ax.set_title("Composition des poids le long de la frontière (long-only)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=N_ASSETS)
    _save(fig, "03_composition_poids_frontiere.png")


def fig04_shrinkage_effect() -> None:
    R, _, _ = simulate_returns(n_assets=8, n_days=150, seed=21)

    def corr_from_cov(C: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.diag(C))
        return C / np.outer(d, d)

    # ledoit_wolf_cov travaille sur rendements bruts → annualiser pour comparer
    S_ann = sample_covariance(R, annualize=True)
    LW = ledoit_wolf_cov(R) * 252
    eig_s = np.sort(np.linalg.eigvalsh(S_ann))[::-1]
    eig_lw = np.sort(np.linalg.eigvalsh(LW))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    im0 = axes[0].imshow(corr_from_cov(S_ann), cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title(r"Corrélation (échantillon $\hat\Sigma$)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(corr_from_cov(LW), cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title(r"Corrélation (Ledoit–Wolf)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[2].semilogy(np.arange(1, len(eig_s) + 1), eig_s, "o-", color="#1f4e79", label="échantillon")
    axes[2].semilogy(np.arange(1, len(eig_lw) + 1), eig_lw, "s-", color="#e67e22", label="shrinkage")
    axes[2].set_xlabel("Rang de la valeur propre")
    axes[2].set_ylabel(r"$\lambda_k$ (échelle log)")
    axes[2].set_title(r"Spectre de $\Sigma$")
    axes[2].legend()
    fig.suptitle("Effet du shrinkage sur la matrice de covariance", y=1.02)
    _save(fig, "04_effet_shrinkage_covariance.png")


def fig05_backtest_equity() -> None:
    R, mu, cov = simulate_returns(n_assets=N_ASSETS, n_days=756, seed=SEED)
    # estimateurs sur fenêtre train
    split = 504
    R_train, R_test = R[:split], R[split:]
    mu_hat = R_train.mean(axis=0) * 252
    cov_hat = np.cov(R_train, rowvar=False) * 252
    cov_lw = ledoit_wolf_cov(R_train) * 252

    w_eq = np.ones(N_ASSETS) / N_ASSETS
    w_mv = optimize_long_only(mu_hat, cov_hat, target_return=float(np.mean(mu_hat)))
    w_lw = optimize_long_only(mu_hat, cov_lw, target_return=float(np.mean(mu_hat)))

    eq_eq = backtest_equity(R_test, w_eq, rebalance_every=21)
    eq_mv = backtest_equity(R_test, w_mv, rebalance_every=21)
    eq_lw = backtest_equity(R_test, w_lw, rebalance_every=21)
    days = np.arange(len(eq_eq))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(days, eq_eq, color="#7a8b99", lw=1.8, label="Égal-pondéré")
    ax.plot(days, eq_mv, color="#1f4e79", lw=2.0, label="Markowitz (Σ échantillon)")
    ax.plot(days, eq_lw, color="#e67e22", lw=2.0, label="Markowitz (Σ shrinkage)")
    ax.set_xlabel("Jours (période out-of-sample)")
    ax.set_ylabel("Richesse (base 1)")
    ax.set_title("Backtest : courbes d'équité (rebalancement mensuel)")
    ax.legend(loc="best")
    _save(fig, "05_backtest_equity_curve.png")


def fig06_sharpe_along_frontier() -> None:
    _, mu, cov = simulate_returns(n_assets=N_ASSETS, seed=SEED)
    t, v, W = efficient_frontier_long_only(mu, cov, n_points=35)
    s = np.array([sharpe_ratio(ti, vi, rf=0.02) for ti, vi in zip(t, v)])
    i_star = int(np.nanargmax(s))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(v, s, color="#1f4e79", lw=2.0)
    ax.scatter([v[i_star]], [s[i_star]], s=90, c="#c0392b", zorder=5, label=f"Max Sharpe ≈ {s[i_star]:.2f}")
    ax.set_xlabel(r"Risque $\sigma_p$")
    ax.set_ylabel("Ratio de Sharpe ($r_f=2\\%$)")
    ax.set_title("Sharpe le long de la frontière long-only")
    ax.legend(loc="best")
    _save(fig, "06_sharpe_le_long_frontiere.png")


def fig07_concept_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.4, 3.2, 2.2, 1.2, "Données\nrendements $R$"),
        (3.2, 3.2, 2.4, 1.2, r"Estimateurs" + "\n" + r"$\hat\mu,\hat\Sigma$"),
        (6.2, 3.2, 2.6, 1.2, "Optimisation\nQP / Lagrange"),
        (3.2, 0.6, 2.4, 1.2, "Portefeuille $w^*$\nfrontière / Sharpe"),
        (6.2, 0.6, 2.6, 1.2, "Évaluation\nbacktest, Sharpe"),
    ]
    for x, y, w, h, txt in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor="#e8eef4", edgecolor="#1f4e79", lw=1.5,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                color="#333333", lw=1.3,
            )
        )

    arrow(2.6, 3.8, 3.2, 3.8)
    arrow(5.6, 3.8, 6.2, 3.8)
    arrow(7.5, 3.2, 4.4, 1.8)
    arrow(5.6, 1.2, 6.2, 1.2)
    ax.set_title("Schéma conceptuel du pipeline Markowitz", pad=8)
    _save(fig, "07_schema_conceptuel_modele.png")


def fig08_condition_number_vs_n() -> None:
    """Conditionnement de Σ̂ en fonction du ratio p/n (nombre d'actifs / jours)."""
    rng_seeds = range(5)
    n_days_list = [60, 90, 120, 180, 252, 504]
    p = 10
    cond_s, cond_lw = [], []
    for n_days in n_days_list:
        cs, cl = [], []
        for s in rng_seeds:
            R, _, _ = simulate_returns(n_assets=p, n_days=n_days, seed=s)
            S = np.cov(R, rowvar=False)
            LW = ledoit_wolf_cov(R)
            cs.append(np.linalg.cond(S))
            cl.append(np.linalg.cond(LW))
        cond_s.append(np.median(cs))
        cond_lw.append(np.median(cl))

    ratio = p / np.array(n_days_list, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(ratio, cond_s, "o-", color="#1f4e79", label=r"cond$(\hat\Sigma)$ échantillon")
    ax.semilogy(ratio, cond_lw, "s-", color="#e67e22", label=r"cond$(\Sigma_{LW})$")
    ax.set_xlabel(r"Ratio $p/n$ (actifs / observations)")
    ax.set_ylabel("Nombre de conditionnement")
    ax.set_title("Stabilité numérique : shrinkage vs échantillon")
    ax.legend(loc="best")
    _save(fig, "08_conditionnement_vs_ratio_pn.png")


def fig09_weights_analytic_vs_long_only() -> None:
    _, mu, cov = simulate_returns(n_assets=N_ASSETS, seed=SEED)
    target = float(np.median(mu))
    # poids analytique au même rendement
    t_a, _, W_a = efficient_frontier_analytic(mu, cov, n_points=80)
    i = int(np.argmin(np.abs(t_a - target)))
    w_a = W_a[i]
    w_l = optimize_long_only(mu, cov, target_return=target)

    x = np.arange(N_ASSETS)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width / 2, w_a, width, color="#1f4e79", label="Analytique (short ok)")
    ax.bar(x + width / 2, w_l, width, color="#e67e22", label="Long-only (QP)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Poids")
    ax.set_title(fr"Poids pour $\mu_p \approx {target:.3f}$ : short autorisé vs long-only")
    ax.legend(loc="best")
    _save(fig, "09_poids_analytique_vs_long_only.png")


def fig10_gmv_and_max_sharpe() -> None:
    _, mu, cov = simulate_returns(n_assets=N_ASSETS, seed=SEED)
    t, v, _ = efficient_frontier_long_only(mu, cov, n_points=40)
    w_gmv = gmv_weights(cov, long_only=True)
    w_ms = max_sharpe_weights(mu, cov, rf=0.02, long_only=True)
    r_g, v_g = portfolio_stats(w_gmv, mu, cov)
    r_m, v_m = portfolio_stats(w_ms, mu, cov)
    vols = np.sqrt(np.diag(cov))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(v, t, color="#1f4e79", lw=2.0, label="Frontière long-only")
    ax.scatter(vols, mu, s=50, c="#95a5a6", label="Actifs")
    ax.scatter([v_g], [r_g], s=110, c="#27ae60", zorder=5, label="GMV")
    ax.scatter([v_m], [r_m], s=110, c="#c0392b", zorder=5, label="Max Sharpe")
    ax.set_xlabel(r"Risque $\sigma_p$")
    ax.set_ylabel(r"Rendement $\mu_p$")
    ax.set_title("Portefeuilles remarquables sur la frontière")
    ax.legend(loc="best")
    _save(fig, "10_gmv_et_max_sharpe.png")


def main() -> None:
    print("Génération des figures Markowitz…")
    fig01_risk_return_cloud()
    fig02_frontier_analytic_vs_long_only()
    fig03_weights_along_frontier()
    fig04_shrinkage_effect()
    fig05_backtest_equity()
    fig06_sharpe_along_frontier()
    fig07_concept_diagram()
    fig08_condition_number_vs_n()
    fig09_weights_analytic_vs_long_only()
    fig10_gmv_and_max_sharpe()
    print(f"Terminé : {len(list(OUT.glob('*.png')))} PNG dans {OUT}")


if __name__ == "__main__":
    main()
