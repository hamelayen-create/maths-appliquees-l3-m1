#!/usr/bin/env python3
"""Génère les figures du rapport 09-crypto-info."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crypto_info import (  # noqa: E402
    H,
    binary_entropy,
    bsc_capacity,
    generate_rsa_keypair,
    hamming74_decode,
    hamming74_encode,
    rsa_decrypt,
    rsa_encrypt,
    simulate_hamming_ber,
)

FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
    }
)


def fig01_rsa_schema() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Schéma de principe RSA (clés, chiffrement, déchiffrement)")

    boxes = [
        (0.4, 2.8, 2.4, 1.4, "Alice\nclé publique\n(n, e)"),
        (3.4, 2.8, 2.4, 1.4, "Message m\n0 ≤ m < n"),
        (6.4, 2.8, 2.6, 1.4, "Chiffre\nc ≡ mᵉ (mod n)"),
        (9.4, 2.8, 2.2, 1.4, "Bob\nclé privée d"),
    ]
    colors = ["#dbeafe", "#fef3c7", "#fce7f3", "#dcfce7"]
    for (x, y, w, h, txt), col in zip(boxes, colors):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                facecolor=col, edgecolor="#334155", linewidth=1.4,
            )
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)

    ax.annotate("", xy=(3.3, 3.5), xytext=(2.9, 3.5),
                arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.6))
    ax.annotate("", xy=(6.3, 3.5), xytext=(5.9, 3.5),
                arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.6))
    ax.annotate("", xy=(9.3, 3.5), xytext=(9.1, 3.5),
                arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.6))

    ax.add_patch(
        FancyBboxPatch(
            (3.0, 0.5), 6.0, 1.5, boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#f1f5f9", edgecolor="#334155", linewidth=1.2,
        )
    )
    ax.text(
        6.0, 1.25,
        "Génération : p, q premiers → n = pq, φ = (p−1)(q−1)\n"
        "e premier avec φ, d ≡ e⁻¹ (mod φ)\n"
        "Vérification : m ≡ cᵈ ≡ mᵉᵈ ≡ m (mod n)  [petit théorème / Euler]",
        ha="center", va="center", fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig01_rsa_schema.png", bbox_inches="tight")
    plt.close(fig)


def fig02_entropy_capacity() -> None:
    p = np.linspace(1e-4, 1 - 1e-4, 400)
    h = np.array([binary_entropy(x) for x in p])
    c = np.array([bsc_capacity(x) for x in p])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(p, h, color="#b45309", lw=2.2, label=r"$h(p)=H(\mathrm{Bernoulli}(p))$")
    ax.plot(p, c, color="#0369a1", lw=2.2, label=r"$C_{\mathrm{BSC}}(p)=1-h(p)$")
    ax.axvline(0.5, color="#94a3b8", ls="--", lw=1)
    ax.set_xlabel(r"Probabilité d'erreur $p$")
    ax.set_ylabel("bits")
    ax.set_title("Entropie binaire $H(p)$ et capacité du canal BSC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig02_entropy_bsc_capacity.png", bbox_inches="tight")
    plt.close(fig)


def fig03_hamming_structure() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ax.set_title("Structure Hamming(7,4) : bits de données et de parité")

    labels = ["p1", "p2", "d1", "p4", "d2", "d3", "d4"]
    roles = ["parité", "parité", "data", "parité", "data", "data", "data"]
    colors = ["#fecaca" if r == "parité" else "#bbf7d0" for r in roles]
    for i, (lab, role, col) in enumerate(zip(labels, roles, colors), start=1):
        x = 0.4 + (i - 1) * 1.05
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.2), 0.9, 1.2, boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=col, edgecolor="#1e293b", linewidth=1.3,
            )
        )
        ax.text(x + 0.45, 1.95, lab, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + 0.45, 1.45, f"pos {i}", ha="center", va="center", fontsize=9)
    ax.text(4, 0.55, "Positions de parité : 1, 2, 4  ·  Positions de données : 3, 5, 6, 7",
            ha="center", fontsize=10)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#fecaca", edgecolor="#1e293b", label="Parité"),
            mpatches.Patch(facecolor="#bbf7d0", edgecolor="#1e293b", label="Données"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig03_hamming_structure.png", bbox_inches="tight")
    plt.close(fig)


def fig04_syndrome_map() -> None:
    data = np.array([1, 0, 1, 1])
    code = hamming74_encode(data)
    positions = list(range(8))  # 0 = no error
    syndromes = []
    for pos in positions:
        recv = code.copy()
        if pos > 0:
            recv[pos - 1] ^= 1
        syn = (H @ recv) % 2
        syndromes.append(syn)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    mat = np.array(syndromes)
    im = ax.imshow(mat.T, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["s0", "s1", "s2"])
    ax.set_xticks(range(8))
    ax.set_xticklabels(["0\n(ok)"] + [str(i) for i in range(1, 8)])
    ax.set_xlabel("Position d'erreur (1-based, 0 = aucune)")
    ax.set_ylabel("Bits du syndrome")
    ax.set_title("Syndrome → position d'erreur (Hamming 7,4)")
    for i in range(8):
        for j in range(3):
            ax.text(i, j, str(int(mat[i, j])), ha="center", va="center", color="#0f172a")
    # annotation: binary value of syndrome equals position
    for i in range(8):
        val = int(mat[i, 0] + 2 * mat[i, 1] + 4 * mat[i, 2])
        ax.text(i, 3.35, f"={val}", ha="center", va="bottom", fontsize=9, color="#334155")
    ax.set_ylim(2.5, -0.5)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="valeur bit")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig04_syndrome_position.png", bbox_inches="tight")
    plt.close(fig)


def fig05_ber_before_after() -> None:
    ps = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15])
    raw, corr = [], []
    for p in ps:
        r, c = simulate_hamming_ber(float(p), n_blocks=8000, seed=11)
        raw.append(r)
        corr.append(c)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(ps, raw, "o-", color="#b91c1c", lw=2, label="BER données (sans correction)")
    ax.plot(ps, corr, "s-", color="#15803d", lw=2, label="BER données (après Hamming)")
    ax.set_xlabel(r"Probabilité d'erreur canal $p$")
    ax.set_ylabel("Taux d'erreur bit (données)")
    ax.set_title("Taux d'erreur avant / après correction Hamming(7,4)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig05_ber_before_after.png", bbox_inches="tight")
    plt.close(fig)


def fig06_comm_model() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_title("Bloc-diagramme du modèle : source → crypto/codage → canal → décodage")

    items = [
        (0.3, "Source\nbits / message"),
        (3.0, "RSA\n(optionnel)"),
        (5.7, "Encodeur\nHamming"),
        (8.5, "Canal\nBSC(p)"),
        (11.3, "Décodeur\n+ lecture"),
    ]
    for x, txt in items:
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.1), 2.2, 1.4, boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor="#e2e8f0", edgecolor="#1e293b", linewidth=1.3,
            )
        )
        ax.text(x + 1.1, 1.8, txt, ha="center", va="center", fontsize=10)
    for x0 in [2.5, 5.2, 7.9, 10.7]:
        ax.annotate("", xy=(x0 + 0.45, 1.8), xytext=(x0, 1.8),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#0f172a"))
    ax.text(7.0, 0.45, "Objectif : confidentialité (RSA pédagogique) + fiabilité (Hamming) sur un canal bruité",
            ha="center", fontsize=9.5, style="italic")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig06_modele_communication.png", bbox_inches="tight")
    plt.close(fig)


def fig07_rsa_roundtrip_demo() -> None:
    keys = generate_rsa_keypair(bits=16, seed=42)
    messages = np.arange(0, min(40, keys.n), dtype=int)
    ciphers = np.array([rsa_encrypt(int(m), keys.n, keys.e) for m in messages])
    recovered = np.array([rsa_decrypt(int(c), keys.n, keys.d) for c in ciphers])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].scatter(messages, ciphers, s=28, c="#7c2d12", alpha=0.85)
    axes[0].set_xlabel("message m")
    axes[0].set_ylabel("chiffre c")
    axes[0].set_title(f"RSA pédagogique : m → c  (n={keys.n})")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(messages, recovered, s=28, c="#14532d", alpha=0.85)
    axes[1].plot([0, messages.max()], [0, messages.max()], "--", color="#94a3b8", lw=1)
    axes[1].set_xlabel("message m")
    axes[1].set_ylabel("déchiffré")
    axes[1].set_title("Vérification : déchiffrement = identité")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Expérience RSA (petites clés — usage éducatif uniquement)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig07_rsa_roundtrip.png", bbox_inches="tight")
    plt.close(fig)


def fig08_rate_vs_capacity() -> None:
    p = np.linspace(0.001, 0.25, 200)
    c = np.array([bsc_capacity(x) for x in p])
    rate_hamming = 4 / 7

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(p, c, color="#0369a1", lw=2.2, label=r"$C_{\mathrm{BSC}}(p)$")
    ax.axhline(rate_hamming, color="#b45309", lw=2, ls="--", label=r"taux Hamming $R=4/7$")
    # region where rate > capacity
    ax.fill_between(
        p, c, rate_hamming, where=(rate_hamming > c),
        color="#fecaca", alpha=0.45, label="R > C (fiabilité asymptotique impossible)",
    )
    ax.set_xlabel(r"$p$")
    ax.set_ylabel("bits / usage canal")
    ax.set_title("Taux du code Hamming vs capacité BSC")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig08_rate_vs_capacity.png", bbox_inches="tight")
    plt.close(fig)


def fig09_parity_checks() -> None:
    """Visualise les trois équations de parité comme ensembles de positions."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    checks = [
        ("Parité p1 (ligne s0)", [1, 3, 5, 7], "#fecaca"),
        ("Parité p2 (ligne s1)", [2, 3, 6, 7], "#bfdbfe"),
        ("Parité p4 (ligne s2)", [4, 5, 6, 7], "#bbf7d0"),
    ]
    for ax, (title, positions, col) in zip(axes, checks):
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 2.5)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
        for i in range(1, 8):
            face = col if i in positions else "#f8fafc"
            edge = "#0f172a" if i in positions else "#94a3b8"
            ax.add_patch(
                FancyBboxPatch(
                    (i - 0.35, 0.7), 0.7, 0.9,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=face, edgecolor=edge, linewidth=1.2,
                )
            )
            ax.text(i, 1.15, str(i), ha="center", va="center", fontsize=11)
        ax.text(4, 0.25, "somme modulo 2 = 0", ha="center", fontsize=9, color="#334155")
    fig.suptitle("Équations de contrôle Hamming(7,4)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig09_parity_checks.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig01_rsa_schema()
    fig02_entropy_capacity()
    fig03_hamming_structure()
    fig04_syndrome_map()
    fig05_ber_before_after()
    fig06_comm_model()
    fig07_rsa_roundtrip_demo()
    fig08_rate_vs_capacity()
    fig09_parity_checks()
    pngs = sorted(FIGDIR.glob("*.png"))
    print(f"Generated {len(pngs)} figures in {FIGDIR}")
    for p in pngs:
        print(" -", p.name)


if __name__ == "__main__":
    main()
