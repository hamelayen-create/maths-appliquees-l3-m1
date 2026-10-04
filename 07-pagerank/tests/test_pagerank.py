import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pagerank import pagerank_linear, pagerank_power, random_graph


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
