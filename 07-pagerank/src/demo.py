#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pagerank import fiedler_vector, pagerank_linear, pagerank_power, random_graph


def tiny_graph() -> sparse.csr_matrix:
    # 0→1→2→0, et 2→3
    rows = [0, 1, 2, 2]
    cols = [1, 2, 0, 3]
    data = [1, 1, 1, 1]
    return sparse.csr_matrix((data, (rows, cols)), shape=(4, 4), dtype=float)


def main() -> None:
    print("=== PageRank ===")
    adj = tiny_graph()
    r_pow = pagerank_power(adj)
    r_lin = pagerank_linear(adj)
    print("tiny graph power :", np.round(r_pow, 4))
    print("tiny graph linear:", np.round(r_lin, 4))
    print("L1 gap =", float(np.linalg.norm(r_pow - r_lin, 1)))

    g = random_graph(80, p=0.05, seed=3)
    r = pagerank_power(g)
    top = np.argsort(-r)[:5]
    print("top-5 nodes (random digraph):", list(zip(top.tolist(), np.round(r[top], 4).tolist())))

    # Fiedler sur version non orientée
    und = g.maximum(g.T)
    f = fiedler_vector(und)
    print("Fiedler sign split sizes:", int(np.sum(f >= 0)), int(np.sum(f < 0)))


if __name__ == "__main__":
    main()
