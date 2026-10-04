#!/usr/bin/env python3
"""Génère les figures du rapport 07-pagerank (PNG dans report/figures/)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pagerank import (  # noqa: E402
    fiedler_vector,
    pagerank_linear,
    pagerank_power,
    random_graph,
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


def tiny_graph() -> sparse.csr_matrix:
    rows = [0, 1, 2, 2]
    cols = [1, 2, 0, 3]
    return sparse.csr_matrix((np.ones(4), (rows, cols)), shape=(4, 4))


def fig01_annotated_scores() -> None:
    """Petit graphe annoté avec scores PageRank."""
    adj = tiny_graph()
    r = pagerank_power(adj, alpha=0.85, tol=1e-12)
    pos = {
        0: (0.15, 0.55),
        1: (0.5, 0.85),
        2: (0.85, 0.55),
        3: (0.5, 0.15),
    }
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    for u, v in edges:
        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops=dict(arrowstyle="->", color="#334155", lw=1.8, shrinkA=18, shrinkB=18),
        )
    sizes = 900 + 3200 * r
    for i, (x, y) in pos.items():
        ax.scatter([x], [y], s=sizes[i], c=[r[i]], cmap="YlOrRd", vmin=r.min() * 0.9, vmax=r.max(), zorder=3, edgecolors="#1e293b", linewidths=1.2)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=13, fontweight="bold", color="#0f172a", zorder=4)
        ax.text(x, y - 0.11, f"r={r[i]:.3f}", ha="center", va="top", fontsize=10, color="#334155")
    ax.set_title(r"Graphe jouet et scores PageRank ($\alpha=0.85$)")
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=r.min(), vmax=r.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("score $r_i$")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig01_graphe_annote.png")
    plt.close(fig)


def fig02_convergence() -> None:
    """Convergence ||r_{k+1}-r_k||_1."""
    g = random_graph(80, p=0.05, seed=3)
    _, hist = pagerank_power(g, alpha=0.85, tol=1e-14, max_iter=120, return_history=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(np.arange(1, len(hist) + 1), hist, color="#0f766e", lw=2)
    ax.set_xlabel("itération $k$")
    ax.set_ylabel(r"$\|r_{k+1}-r_k\|_1$")
    ax.set_title("Convergence de la méthode de la puissance")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig02_convergence.png")
    plt.close(fig)


def fig03_damping() -> None:
    """Effet du facteur d'amortissement α."""
    g = random_graph(60, p=0.07, seed=2)
    alphas = np.linspace(0.5, 0.99, 12)
    entropies = []
    max_scores = []
    for a in alphas:
        r = pagerank_power(g, alpha=float(a), tol=1e-12)
        r = np.clip(r, 1e-30, None)
        entropies.append(float(-(r * np.log(r)).sum()))
        max_scores.append(float(r.max()))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(alphas, max_scores, "o-", color="#b45309", label=r"$\max_i r_i$")
    ax1.set_xlabel(r"damping $\alpha$")
    ax1.set_ylabel(r"$\max_i r_i$", color="#b45309")
    ax2 = ax1.twinx()
    ax2.plot(alphas, entropies, "s--", color="#1d4ed8", label="entropie")
    ax2.set_ylabel(r"entropie $-\sum r_i\log r_i$", color="#1d4ed8")
    ax1.set_title(r"Effet de $\alpha$ : concentration vs uniformité")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig03_effet_alpha.png")
    plt.close(fig)


def fig04_topk() -> None:
    """Top-k nœuds sur graphe aléatoire."""
    g = random_graph(80, p=0.05, seed=3)
    r = pagerank_power(g, alpha=0.85, tol=1e-12)
    k = 12
    top = np.argsort(-r)[:k]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(np.arange(k), r[top], color="#0e7490", edgecolor="#134e4a")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([str(i) for i in top], rotation=0)
    ax.set_xlabel("indice du nœud")
    ax.set_ylabel("score PageRank")
    ax.set_title(f"Top-{k} nœuds (Erdős–Rényi orienté, n=80, p=0.05)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig04_topk.png")
    plt.close(fig)


def fig05_fiedler() -> None:
    """Partition induite par le vecteur de Fiedler."""
    # deux clusters reliés par un pont
    n = 24
    rows, cols = [], []
    for block, offset in [(range(12), 0), (range(12, 24), 12)]:
        for i in block:
            for j in block:
                if i != j and np.random.default_rng(i * 31 + j).random() < 0.45:
                    rows.append(i)
                    cols.append(j)
    rows += [5, 17]
    cols += [17, 5]
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    und = adj.maximum(adj.T)
    f = fiedler_vector(und)
    # layout: deux cercles
    pos = {}
    for i in range(12):
        th = 2 * np.pi * i / 12
        pos[i] = (np.cos(th) - 1.6, np.sin(th))
    for i in range(12, 24):
        th = 2 * np.pi * (i - 12) / 12
        pos[i] = (np.cos(th) + 1.6, np.sin(th))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    # gauche: graphe coloré
    ax = axes[0]
    A = und.tocoo()
    for i, j, v in zip(A.row, A.col, A.data):
        if i < j:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#94a3b8", lw=0.7, zorder=1)
    colors = ["#dc2626" if fi >= 0 else "#2563eb" for fi in f]
    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, c=colors, s=70, zorder=2, edgecolors="#0f172a", linewidths=0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Partition par signe de Fiedler")
    # droite: composantes du vecteur
    ax = axes[1]
    order = np.argsort(f)
    ax.stem(np.arange(n), f[order], linefmt="#475569", markerfmt="o", basefmt=" ")
    ax.axhline(0, color="#0f172a", lw=1)
    ax.set_xlabel("nœuds triés par $f_i$")
    ax.set_ylabel("composante de Fiedler $f_i$")
    ax.set_title("Vecteur de Fiedler (trié)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig05_fiedler.png")
    plt.close(fig)


def fig06_schema_conceptuel() -> None:
    """Schéma conceptuel du modèle du surfeur aléatoire."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15", facecolor=color, edgecolor="#1e293b", lw=1.3)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)

    box(0.3, 3.2, 2.4, 1.2, "Graphe orienté\n$A$ (adjacence)", "#fef3c7")
    box(3.5, 3.2, 2.6, 1.2, "Transition $P$\n+ dangling", "#dbeafe")
    box(6.8, 3.2, 2.8, 1.2, "Google matrix\n$G=\\alpha P+(1-\\alpha)E$", "#dcfce7")
    box(2.0, 0.7, 2.8, 1.2, "Itération puissance\n$r\\leftarrow G^\\top r$", "#fce7f3")
    box(5.5, 0.7, 3.0, 1.2, "Système linéaire\n$(I-\\alpha P^\\top)r=v$", "#e0e7ff")

    arrows = [
        ((2.7, 3.8), (3.5, 3.8)),
        ((6.1, 3.8), (6.8, 3.8)),
        ((8.2, 3.2), (7.0, 1.9)),
        ((8.0, 3.2), (3.4, 1.9)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#334155", lw=1.6))
    ax.text(5, 4.7, "Pipeline PageRank : du graphe au score", ha="center", fontsize=13, fontweight="bold")
    ax.text(5, 0.25, r"téléportation uniforme $v=(1-\alpha)/n\cdot\mathbf{1}$", ha="center", fontsize=10, color="#475569")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig06_schema_modele.png")
    plt.close(fig)


def fig07_power_vs_linear() -> None:
    """Écart L1 power vs linear selon n."""
    ns = [20, 40, 60, 80, 100]
    gaps = []
    for n in ns:
        g = random_graph(n, p=0.08, seed=5)
        r1 = pagerank_power(g, tol=1e-12)
        r2 = pagerank_linear(g)
        gaps.append(float(np.linalg.norm(r1 - r2, 1)))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(ns, gaps, "o-", color="#7c3aed", lw=2)
    ax.set_xlabel("taille $n$")
    ax.set_ylabel(r"$\|r_{\mathrm{power}}-r_{\mathrm{linear}}\|_1$")
    ax.set_title("Accord méthode de la puissance / solveur linéaire")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig07_power_vs_linear.png")
    plt.close(fig)


def fig08_dangling() -> None:
    """Influence des dangling nodes sur la masse de probabilité."""
    # chaîne + un dangling
    n = 8
    rows = list(range(n - 2))
    cols = list(range(1, n - 1))
    # nœud n-1 isolé en sortie (dangling), reçoit depuis n-2
    rows.append(n - 2)
    cols.append(n - 1)
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    r = pagerank_power(adj, alpha=0.85, tol=1e-12)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = ["#0e7490"] * (n - 1) + ["#dc2626"]
    ax.bar(np.arange(n), r, color=colors, edgecolor="#134e4a")
    ax.set_xlabel("nœud")
    ax.set_ylabel("score PageRank")
    ax.set_title("Chaîne avec dangling node (en rouge : nœud sans sortie)")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#0e7490"),
            plt.Rectangle((0, 0), 1, 1, color="#dc2626"),
        ],
        labels=["nœud standard", "dangling"],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig08_dangling.png")
    plt.close(fig)


def fig09_sparsity() -> None:
    """Motif de sparsité de A et de la Google matrix dense effective."""
    g = random_graph(40, p=0.08, seed=1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    axes[0].spy(g, markersize=3, color="#0f766e")
    axes[0].set_title("Adjacence sparse $A$")
    axes[0].grid(False)
    # Google matrix dense (illustrative)
    P = g.tocsr().astype(float)
    out = np.asarray(P.sum(axis=1)).ravel()
    dangling = out == 0
    inv = np.zeros_like(out)
    inv[~dangling] = 1.0 / out[~dangling]
    P = sparse.diags(inv) @ P
    if dangling.any():
        rows = np.flatnonzero(dangling)
        data = np.ones(len(rows) * P.shape[0]) / P.shape[0]
        row_idx = np.repeat(rows, P.shape[0])
        col_idx = np.tile(np.arange(P.shape[0]), len(rows))
        P = P + sparse.csr_matrix((data, (row_idx, col_idx)), shape=P.shape)
    alpha = 0.85
    G = alpha * P.toarray() + (1 - alpha) / P.shape[0]
    im = axes[1].imshow(G, cmap="viridis", aspect="auto")
    axes[1].set_title(r"Google matrix $G$ (dense)")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig09_sparsity.png")
    plt.close(fig)


def fig10_score_distribution() -> None:
    """Distribution empirique des scores sur graphe aléatoire."""
    g = random_graph(120, p=0.04, seed=7)
    r = pagerank_power(g, alpha=0.85, tol=1e-12)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(r, bins=25, color="#0369a1", edgecolor="white", alpha=0.9)
    ax.axvline(1 / g.shape[0], color="#b45309", ls="--", lw=2, label=r"$1/n$")
    ax.set_xlabel("score $r_i$")
    ax.set_ylabel("effectif")
    ax.set_title("Distribution des scores (n=120)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig10_distribution.png")
    plt.close(fig)


def main() -> None:
    fig01_annotated_scores()
    fig02_convergence()
    fig03_damping()
    fig04_topk()
    fig05_fiedler()
    fig06_schema_conceptuel()
    fig07_power_vs_linear()
    fig08_dangling()
    fig09_sparsity()
    fig10_score_distribution()
    print(f"Figures écrites dans {FIGDIR}")
    for p in sorted(FIGDIR.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
