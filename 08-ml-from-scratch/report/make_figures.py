#!/usr/bin/env python3
"""Génère les figures du rapport 08-ml-from-scratch."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml import (  # noqa: E402
    LinearRegression,
    LinearSVM,
    PCA,
    make_blobs,
    make_regression,
    mse,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Style sobre, reproductible
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


def fig01_regression_fit_residuals() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(-2.5, 2.5, 80)
    y = 1.2 * x + 0.4 + 0.35 * rng.normal(size=len(x))
    X = x.reshape(-1, 1)
    model = LinearRegression(ridge=0.0).fit_normal(X, y)
    yhat = model.predict(X)
    resid = y - yhat

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].scatter(x, y, s=28, alpha=0.75, color="#1f4e79", label="données")
    xx = np.linspace(-2.7, 2.7, 200)
    axes[0].plot(
        xx,
        model.predict(xx.reshape(-1, 1)),
        color="#c0392b",
        lw=2.2,
        label=rf"fit: $\hat y={model.coef_[0]:.2f}x+{model.intercept_:.2f}$",
    )
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title("Régression linéaire : ajustement")
    axes[0].legend(loc="best", framealpha=0.9)

    axes[1].axhline(0.0, color="black", lw=1)
    axes[1].scatter(yhat, resid, s=28, alpha=0.75, color="#1f4e79")
    axes[1].set_xlabel(r"$\hat y$")
    axes[1].set_ylabel(r"$y-\hat y$")
    axes[1].set_title(f"Résidus (MSE={mse(y, yhat):.3f})")

    fig.tight_layout()
    fig.savefig(FIGDIR / "01_regression_fit_residus.png")
    plt.close(fig)


def fig02_gradient_path() -> None:
    # Problème 1D pour visualiser le chemin dans le plan (b, w)
    rng = np.random.default_rng(1)
    x = rng.normal(size=120)
    y = 0.8 * x + 0.3 + 0.25 * rng.normal(size=120)
    X = x.reshape(-1, 1)

    model = LinearRegression(ridge=0.05).fit_gd(
        X, y, lr=0.15, n_iter=80, store_path=True, path_every=1
    )
    path = np.asarray(model.path_)
    opt = LinearRegression(ridge=0.05).fit_normal(X, y)
    b_star, w_star = opt.intercept_, float(opt.coef_[0])

    # Grille de niveau de la perte ridge
    B, W = np.meshgrid(
        np.linspace(b_star - 1.2, b_star + 1.2, 120),
        np.linspace(w_star - 1.5, w_star + 1.5, 120),
    )
    loss = np.zeros_like(B)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            pred = W[i, j] * x + B[i, j]
            err = pred - y
            loss[i, j] = 0.5 * np.mean(err**2) + 0.5 * 0.05 * W[i, j] ** 2

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    cs = axes[0].contour(B, W, loss, levels=18, cmap="viridis")
    axes[0].clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    axes[0].plot(path[:, 0], path[:, 1], "o-", color="#c0392b", ms=3.5, lw=1.4, label="chemin GD")
    axes[0].scatter([b_star], [w_star], c="black", s=60, zorder=5, label="optimum (normale)")
    axes[0].set_xlabel("intercept $b$")
    axes[0].set_ylabel("pente $w$")
    axes[0].set_title("Chemin de convergence du gradient")
    axes[0].legend(loc="best")

    axes[1].semilogy(model.loss_history_, color="#1f4e79", lw=2)
    axes[1].set_xlabel("itération")
    axes[1].set_ylabel("perte ridge")
    axes[1].set_title("Décroissance de la fonction objectif")

    fig.tight_layout()
    fig.savefig(FIGDIR / "02_chemin_gradient.png")
    plt.close(fig)


def fig03_pca_variance() -> None:
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 5))
    # Corrélations anisotropes
    scales = np.array([3.0, 1.5, 0.8, 0.3, 0.1])
    X = Z * scales
    X[:, 1] += 0.7 * X[:, 0]
    X[:, 2] += 0.4 * X[:, 0]
    pca = PCA(n_components=5).fit(X)
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    idx = np.arange(1, len(ratios) + 1)
    ax.bar(idx, ratios, color="#1f4e79", alpha=0.85, label="variance individuelle")
    ax.plot(idx, cum, "o-", color="#c0392b", lw=2, label="variance cumulée")
    ax.axhline(0.9, color="gray", ls="--", lw=1, label="seuil 90%")
    ax.set_xticks(idx)
    ax.set_xlabel("composante principale")
    ax.set_ylabel("proportion de variance")
    ax.set_title("PCA : variance expliquée (via SVD)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_pca_variance.png")
    plt.close(fig)


def fig04_pca_projection() -> None:
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(250, 2))
    A = np.array([[2.5, 1.6], [1.6, 1.2]])
    X = Z @ np.linalg.cholesky(A).T + np.array([1.0, -0.5])
    pca = PCA(n_components=2).fit(X)
    Z2 = pca.transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].scatter(X[:, 0], X[:, 1], s=22, alpha=0.7, color="#1f4e79")
    mean = pca.mean_
    for k, color in enumerate(["#c0392b", "#27ae60"]):
        v = pca.components_[k]
        scale = 3.0 * np.sqrt(pca.explained_variance_ratio_[k] * np.trace(np.cov(X.T)))
        axes[0].arrow(
            mean[0],
            mean[1],
            scale * v[0],
            scale * v[1],
            head_width=0.15,
            color=color,
            length_includes_head=True,
            lw=2,
            label=f"PC{k+1}",
        )
    axes[0].set_aspect("equal")
    axes[0].set_title("Données corrélées + axes principaux")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(loc="best")

    axes[1].scatter(Z2[:, 0], Z2[:, 1], s=22, alpha=0.7, color="#1f4e79")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].set_aspect("equal")
    axes[1].set_title("Projection PCA 2D (coordonnées principales)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    fig.tight_layout()
    fig.savefig(FIGDIR / "04_pca_projection.png")
    plt.close(fig)


def fig05_svm_boundary() -> None:
    X, y = make_blobs(n=220, seed=4)
    svm = LinearSVM(C=1.0, lr=0.05, n_iter=5000).fit(X, y)

    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = svm.decision_function(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.contourf(xx, yy, zz, levels=30, cmap="RdBu", alpha=0.35)
    cs = ax.contour(
        xx,
        yy,
        zz,
        levels=[-1, 0, 1],
        colors=["#27ae60", "black", "#27ae60"],
        linewidths=[1.2, 2.2, 1.2],
        linestyles=["--", "-", "--"],
    )
    ax.clabel(cs, fmt={-1: "marge -1", 0: "frontière", 1: "marge +1"}, fontsize=8)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c="#1f4e79", s=28, label="classe -1", edgecolors="white", lw=0.3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#c0392b", s=28, label="classe +1", edgecolors="white", lw=0.3)
    acc = float(np.mean(svm.predict(X) == y))
    ax.set_title(f"SVM soft-margin : frontière de décision (acc={acc:.3f})")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_svm_frontiere.png")
    plt.close(fig)


def fig06_modele_conceptuel() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 1.3, "Données\n$(X, y)$"),
        (2.5, 1.3, "Prétraitement\ncentrage / échelle"),
        (4.7, 1.3, "Modèle\nlinéaire / PCA / SVM"),
        (6.9, 1.3, "Optimisation\nnormale / GD / sous-grad"),
        (8.9, 1.3, "Évaluation\nMSE, $R^2$, accuracy"),
    ]
    for x, y, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            1.7,
            1.5,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#e8eef5",
            edgecolor="#1f4e79",
            lw=1.8,
        )
        ax.add_patch(patch)
        ax.text(x + 0.85, y + 0.75, text, ha="center", va="center", fontsize=9)

    for x0 in [2.0, 4.2, 6.4, 8.6]:
        arr = FancyArrowPatch(
            (x0, 2.05),
            (x0 + 0.45, 2.05),
            arrowstyle="->",
            mutation_scale=14,
            color="#c0392b",
            lw=1.8,
        )
        ax.add_patch(arr)

    ax.text(
        5,
        3.5,
        "Pipeline pédagogique : de la donnée à la décision",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#1f4e79",
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_schema_pipeline.png")
    plt.close(fig)


def fig07_ridge_biais_variance() -> None:
    rng = np.random.default_rng(5)
    n_train, n_test, d = 40, 200, 12
    # Vrai modèle sparse
    w_true = np.zeros(d)
    w_true[:3] = np.array([1.5, -1.0, 0.8])
    X_test = rng.normal(size=(n_test, d))
    y_test = X_test @ w_true + 0.4 * rng.normal(size=n_test)

    lambdas = np.logspace(-3, 2, 18)
    mse_train, mse_test, norms = [], [], []
    for lam in lambdas:
        X = rng.normal(size=(n_train, d))
        y = X @ w_true + 0.4 * rng.normal(size=n_train)
        # moyenne sur plusieurs tirages pour stabiliser
        tr, te, nm = [], [], []
        for s in range(25):
            rs = np.random.default_rng(100 + s)
            X = rs.normal(size=(n_train, d))
            y = X @ w_true + 0.4 * rs.normal(size=n_train)
            m = LinearRegression(ridge=float(lam)).fit_normal(X, y)
            tr.append(mse(y, m.predict(X)))
            te.append(mse(y_test, m.predict(X_test)))
            nm.append(float(np.linalg.norm(m.coef_)))
        mse_train.append(np.mean(tr))
        mse_test.append(np.mean(te))
        norms.append(np.mean(nm))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].semilogx(lambdas, mse_train, "o-", color="#1f4e79", label="MSE train")
    axes[0].semilogx(lambdas, mse_test, "s-", color="#c0392b", label="MSE test")
    axes[0].set_xlabel(r"$\lambda$ (ridge)")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Compromis biais–variance via ridge")
    axes[0].legend(loc="best")

    axes[1].semilogx(lambdas, norms, "o-", color="#27ae60")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$\|\hat w\|_2$")
    axes[1].set_title("Rétractation des coefficients")

    fig.tight_layout()
    fig.savefig(FIGDIR / "07_ridge_biais_variance.png")
    plt.close(fig)


def fig08_hinge_loss() -> None:
    z = np.linspace(-2.5, 2.5, 400)
    hinge = np.maximum(0.0, 1.0 - z)
    zero_one = (z < 0).astype(float)
    square = (1.0 - z) ** 2

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(z, zero_one, color="gray", lw=2, label="0-1 loss")
    ax.plot(z, hinge, color="#c0392b", lw=2.4, label="hinge $(1-z)_+$")
    ax.plot(z, square, color="#1f4e79", lw=1.8, ls="--", label="square $(1-z)^2$")
    ax.axvline(1.0, color="#27ae60", ls=":", lw=1.5, label="marge fonctionnelle $z=1$")
    ax.set_xlabel(r"marge $z = y\,f(x)$")
    ax.set_ylabel("perte")
    ax.set_title("Hinge loss et approximations convexes")
    ax.set_ylim(-0.1, 4.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "08_hinge_loss.png")
    plt.close(fig)


def fig09_comparaison_methodes() -> None:
    X, y, w_true = make_regression(n=300, d=4, noise=0.15, seed=7)
    normal = LinearRegression(ridge=0.1).fit_normal(X, y)
    gd = LinearRegression(ridge=0.1).fit_gd(X, y, lr=0.05, n_iter=8000)

    labels = [f"$w_{i+1}$" for i in range(len(w_true))]
    x = np.arange(len(w_true))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width, w_true, width, label="vrai", color="#7f8c8d")
    ax.bar(x, normal.coef_, width, label="normale", color="#1f4e79")
    ax.bar(x + width, gd.coef_, width, label="GD", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("valeur du coefficient")
    ax.set_title(
        f"Comparaison normale vs GD\n"
        f"MSE normale={mse(y, normal.predict(X)):.4f}, "
        f"MSE GD={mse(y, gd.predict(X)):.4f}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGDIR / "09_comparaison_normale_gd.png")
    plt.close(fig)


def fig10_svm_loss_curve() -> None:
    X, y = make_blobs(n=200, seed=1)
    svm = LinearSVM(C=1.0, lr=0.05, n_iter=4000).fit(X, y)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    iters = np.arange(1, len(svm.loss_history_) + 1) * 50
    iters[0] = 1
    ax.plot(iters, svm.loss_history_, color="#1f4e79", lw=2)
    ax.set_xlabel("itération (sous-gradient stochastique)")
    ax.set_ylabel("objectif primal approx.")
    ax.set_title("SVM : évolution de l'objectif hinge régularisé")
    fig.tight_layout()
    fig.savefig(FIGDIR / "10_svm_objectif.png")
    plt.close(fig)


def main() -> None:
    fig01_regression_fit_residuals()
    fig02_gradient_path()
    fig03_pca_variance()
    fig04_pca_projection()
    fig05_svm_boundary()
    fig06_modele_conceptuel()
    fig07_ridge_biais_variance()
    fig08_hinge_loss()
    fig09_comparaison_methodes()
    fig10_svm_loss_curve()
    print(f"Figures écrites dans {FIGDIR}")
    for p in sorted(FIGDIR.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
