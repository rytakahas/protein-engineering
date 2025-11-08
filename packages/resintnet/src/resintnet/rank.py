
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import scipy.sparse as sp
import torch

from .graph import load_priors, GraphInputs, build_graph
from .prs import prs_centrality
from .models.sage import normalize_adj, score_nodes


def load_node_features(qid: str, emb_dir: Path, msa_dir: Optional[Path] = None) -> np.ndarray:
    X = np.load(emb_dir / f"{qid}.esm2.npy")
    if msa_dir:
        mp = msa_dir / f"{qid}.msa.npy"
        if mp.exists():
            M = np.load(mp)
            if M.shape[0] == X.shape[0]:
                X = np.concatenate([X, M], axis=1)
    return X


def blend_scores(g_logits: np.ndarray, prs: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    g = 1.0 / (1.0 + np.exp(-g_logits))
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    p = (prs - prs.min()) / (prs.max() - prs.min() + 1e-8)
    return alpha * g + (1 - alpha) * p


def rank_one(
    priors_npz: Path,
    emb_dir: Path,
    msa_dir: Optional[Path],
    alpha: float = 0.5,
    edge_mode: str = "threshold",
    contact_thresh: float = 8.0,
    topk: int = 10,
    device: str = "cpu",
) -> Dict:
    priors, bins, mask, meta = load_priors(priors_npz)
    qid = meta["query_id"]
    L = int(meta["L"])

    X = load_node_features(qid, emb_dir, msa_dir)
    assert X.shape[0] == L, f"Feature length mismatch for {qid}: {X.shape[0]} vs {L}"

    gi = GraphInputs(query_id=qid, L=L, node_features=X, priors=priors, bins=bins, mask=mask)
    G = build_graph(gi, edge_mode=edge_mode, contact_thresh=contact_thresh, topk=topk)
    prs = prs_centrality(G)

    n = L
    rows, cols, vals = [], [], []
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        rows += [u, v]
        cols += [v, u]
        vals += [w, w]
    adj = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A_hat = normalize_adj(adj, self_loops=True)
    logits = score_nodes(torch.from_numpy(X).float(), A_hat, hidden=128, device=device).numpy()

    blended = blend_scores(logits, prs, alpha=alpha)
    order = np.argsort(-blended)
    return {
        "query_id": qid,
        "L": L,
        "scores": blended.tolist(),
        "order": order.tolist(),
        "prs": prs.tolist(),
    }
