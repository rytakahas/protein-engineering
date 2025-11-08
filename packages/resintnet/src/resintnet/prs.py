
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx


def laplacian_matrix(G: nx.Graph) -> sp.csr_matrix:
    n = G.number_of_nodes()
    idx_map = {n_i: i for i, n_i in enumerate(G.nodes())}
    rows, cols, data = [], [], []
    deg = np.zeros(n, dtype=float)
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        iu, iv = idx_map[u], idx_map[v]
        rows += [iu, iv]
        cols += [iv, iu]
        data += [-w, -w]
        deg[iu] += w
        deg[iv] += w
    rows += list(range(n))
    cols += list(range(n))
    data += deg.tolist()
    L = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return L


def prs_centrality(G: nx.Graph, epsilon: float = 1e-3) -> np.ndarray:
    L = laplacian_matrix(G).astype(float)
    n = L.shape[0]
    A = L + epsilon * sp.eye(n, format="csr")
    ones = np.ones(n, dtype=float)
    x = spla.spsolve(A, ones)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x
