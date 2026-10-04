"""PageRank sparse + Laplacien spectral (Fiedler)."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, gmres


def random_graph(n: int = 50, p: float = 0.08, seed: int = 0) -> sparse.csr_matrix:
    """Graphe orienté Erdős–Rényi sous forme d'adjacence sparse 0/1."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < p
    np.fill_diagonal(mask, False)
    return sparse.csr_matrix(mask.astype(float))


def _transition_matrix(adj: sparse.spmatrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    adj = adj.tocsr()
    n = adj.shape[0]
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    dangling = out_deg == 0
    # inv degrees (0 pour dangling → sera géré séparément)
    inv = np.zeros(n)
    inv[~dangling] = 1.0 / out_deg[~dangling]
    D_inv = sparse.diags(inv)
    P = D_inv @ adj
    return P.tocsr(), dangling


def pagerank_power(
    adj: sparse.spmatrix,
    alpha: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
    return_history: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[float]]:
    """Méthode de la puissance pour le PageRank Google.

    Parameters
    ----------
    adj : sparse adjacency (0/1), shape (n, n)
    alpha : facteur d'amortissement (damping), typiquement 0.85
    tol : seuil sur ||r_{k+1} - r_k||_1
    max_iter : nombre maximal d'itérations
    return_history : si True, renvoie aussi la liste des résidus L1

    Returns
    -------
    r : vecteur de scores (somme 1)
    history : optionnel, résidus L1 par itération
    """
    P, dangling = _transition_matrix(adj)
    n = P.shape[0]
    r = np.ones(n) / n
    teleport = (1 - alpha) / n
    history: list[float] = []
    for _ in range(max_iter):
        # contribution dangling : masse redistribuée uniformément
        dangling_mass = alpha * r[dangling].sum() / n if dangling.any() else 0.0
        r_new = alpha * (P.T @ r) + dangling_mass + teleport
        resid = float(np.linalg.norm(r_new - r, 1))
        history.append(resid)
        if resid < tol:
            if return_history:
                return r_new, history
            return r_new
        r = r_new
    if return_history:
        return r, history
    return r


def pagerank_linear(
    adj: sparse.spmatrix,
    alpha: float = 0.85,
) -> np.ndarray:
    """Résout (I - α P^T) r = (1-α)/N * 1  avec correction dangling approximée."""
    P, dangling = _transition_matrix(adj)
    n = P.shape[0]
    # Pour simplicité pédagogique : redistribuer les dangling avant résolution
    if dangling.any():
        # ajouter liens uniformes depuis dangling
        rows = np.flatnonzero(dangling)
        data = np.ones(len(rows) * n) / n
        row_idx = np.repeat(rows, n)
        col_idx = np.tile(np.arange(n), len(rows))
        P = P + sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))

    A = sparse.eye(n, format="csr") - alpha * P.T
    b = np.ones(n) * (1 - alpha) / n
    r, info = gmres(A, b, atol=1e-12, restart=50)
    if info != 0:
        # fallback dense
        r = np.linalg.solve(A.toarray(), b)
    r = np.asarray(r, dtype=float)
    r = np.maximum(r, 0)
    r /= r.sum()
    return r


def fiedler_vector(adj_undirected: sparse.spmatrix) -> np.ndarray:
    """Vecteur de Fiedler du Laplacien (graphe non orienté)."""
    A = adj_undirected.maximum(adj_undirected.T)
    A = A.tocsr()
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A
    # 2 plus petites valeurs propres
    vals, vecs = eigs(L.astype(float), k=2, which="SM")
    order = np.argsort(np.real(vals))
    return np.real(vecs[:, order[1]])
