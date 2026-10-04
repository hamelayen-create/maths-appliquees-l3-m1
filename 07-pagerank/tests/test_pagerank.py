import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pagerank import fiedler_vector, pagerank_linear, pagerank_power, random_graph


def test_pagerank_sums_to_one():
    adj = random_graph(40, p=0.1, seed=1)
    r = pagerank_power(adj)
    assert abs(r.sum() - 1.0) < 1e-8
    assert np.all(r >= -1e-15)


def test_power_matches_linear():
    rows, cols = [0, 1, 2, 0], [1, 2, 0, 2]
    adj = sparse.csr_matrix((np.ones(4), (rows, cols)), shape=(3, 3))
    r1 = pagerank_power(adj, tol=1e-12)
    r2 = pagerank_linear(adj)
    assert np.linalg.norm(r1 - r2, 1) < 1e-6


def test_tiny_graph_ranking_order():
    """Résultat du rapport (§7) : sur 0→1→2→0 et 2→3, le nœud 2 domine."""
    rows = [0, 1, 2, 2]
    cols = [1, 2, 0, 3]
    adj = sparse.csr_matrix((np.ones(4), (rows, cols)), shape=(4, 4))
    r = pagerank_power(adj, alpha=0.85, tol=1e-12)
    assert r.argmax() == 2
    # Ordre observé (§7) : r2 > r1 > r0 ≈ r3 (3 est dangling)
    assert r[2] > r[1] > r[0]
    assert abs(r[0] - r[3]) < 1e-8
    assert abs(r.sum() - 1.0) < 1e-10


def test_fiedler_splits_two_clusters():
    """Deux cliques reliées par un pont : Fiedler sépare les clusters."""
    n = 10
    rows, cols = [], []
    for i in range(5):
        for j in range(5):
            if i != j:
                rows.append(i)
                cols.append(j)
    for i in range(5, 10):
        for j in range(5, 10):
            if i != j:
                rows.append(i)
                cols.append(j)
    rows += [4, 5]
    cols += [5, 4]
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    f = fiedler_vector(adj)
    left = set(np.where(f >= 0)[0].tolist())
    right = set(np.where(f < 0)[0].tolist())
    assert left == set(range(5)) or right == set(range(5))
