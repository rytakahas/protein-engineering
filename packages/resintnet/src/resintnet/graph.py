
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx


@dataclass
class GraphInputs:
    query_id: str
    L: int
    node_features: np.ndarray  # (L, C)
    priors: np.ndarray         # (L, L, B)
    bins: np.ndarray           # (B+1,)
    mask: np.ndarray           # (L, L)


def load_priors(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    priors = data["priors"]
    bins   = data["bins"]
    mask   = data["mask"]
    meta   = json.loads(str(data["meta"]))
    return priors, bins, mask, meta


def expected_distance(priors: np.ndarray, bins: np.ndarray) -> np.ndarray:
    centers = 0.5 * (bins[:-1] + bins[1:])
    return (priors * centers[None, None, :]).sum(-1)


def build_graph(
    gi: GraphInputs,
    edge_mode: str = "threshold",
    contact_thresh: float = 8.0,
    topk: int = 10,
    weight_mode: str = "inv_dist",
) -> nx.Graph:
    L = gi.L
    G = nx.Graph()
    for i in range(L):
        G.add_node(i, x=gi.node_features[i])

    ed = expected_distance(gi.priors, gi.bins)
    ed = np.where(gi.mask, ed, np.inf)

    if edge_mode == "threshold":
        sel = (ed < contact_thresh)
        ii, jj = np.where(sel)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if i >= j:
                continue
            if weight_mode == "inv_dist":
                w = float(1.0 / max(ed[i, j], 1e-3))
            elif weight_mode == "prob_lt_thresh":
                bmask = gi.bins[1:] <= contact_thresh
                p = gi.priors[i, j, :][bmask].sum() if bmask.any() else 0.0
                w = float(p)
            else:
                w = 1.0
            G.add_edge(i, j, weight=w)
    elif edge_mode == "topk":
        for i in range(L):
            order = np.argsort(ed[i])
            keep = []
            for j in order:
                if j == i or not np.isfinite(ed[i, j]):
                    continue
                keep.append(j)
                if len(keep) >= topk:
                    break
            for j in keep:
                w = float(1.0 / max(ed[i, j], 1e-3)) if weight_mode == "inv_dist" else 1.0
                u, v = (i, j) if i < j else (j, i)
                G.add_edge(u, v, weight=w)
    else:
        raise ValueError(f"edge_mode={edge_mode} not supported")

    return G
