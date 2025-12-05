import math
import io
import numpy as np
import pandas as pd
import networkx as nx
from Bio.PDB import MMCIFParser, PDBParser
from scipy.spatial.distance import cdist
from scipy import linalg
from typing import Optional, Tuple, List

AA3_TO1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y"
}
AA1_TO3 = {v:k for k,v in AA3_TO1.items()}

def parse_cif_to_residues(cif_path, chain_id: Optional[str]=None) -> pd.DataFrame:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prot", str(cif_path))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    chain = max(chains, key=lambda ch: sum(1 for _ in ch.get_residues())) if chain_id is None else model[chain_id]
    rows = []
    for res in chain.get_residues():
        if "CA" not in res: 
            continue
        atom = res["CA"]
        rows.append({
            "i": len(rows),
            "resseq": int(res.id[1]),
            "resname": res.get_resname(),
            "x": float(atom.coord[0]),
            "y": float(atom.coord[1]),
            "z": float(atom.coord[2]),
        })
    return pd.DataFrame(rows)

def parse_pdb_to_residues(pdb_path, chain_id: Optional[str]=None) -> pd.DataFrame:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", str(pdb_path))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    chain = max(chains, key=lambda ch: sum(1 for _ in ch.get_residues())) if chain_id is None else model[chain_id]
    rows = []
    for res in chain.get_residues():
        if "CA" not in res:
            continue
        atom = res["CA"]
        rows.append({
            "i": len(rows),
            "resseq": int(res.id[1]),
            "resname": res.get_resname(),
            "x": float(atom.coord[0]),
            "y": float(atom.coord[1]),
            "z": float(atom.coord[2]),
        })
    return pd.DataFrame(rows)

def build_graph_from_contacts(res_df: pd.DataFrame, C: np.ndarray, topk: int=6, min_sep: int=2) -> nx.Graph:
    n = len(res_df)
    G = nx.Graph()
    for i, row in res_df.iterrows():
        G.add_node(int(i), **row.to_dict())
    S = C.copy()
    for i in range(n):
        S[i, i] = -np.inf
        for j in range(n):
            if abs(i - j) < min_sep:
                S[i, j] = -np.inf
    for i in range(n):
        nbrs = np.argsort(-S[i])[:topk]
        for j in nbrs:
            if not np.isfinite(S[i, j]):
                continue
            # store distance-like and contact weight
            G.add_edge(int(i), int(j),
                       dist=float(1.0 / max(S[i, j], 1e-6)),
                       w_contact=float(S[i, j]))
    return G

def build_topk_graph(res_df, topk=6, min_sep=2, sigma=8.0) -> nx.Graph:
    coords = res_df[["x","y","z"]].to_numpy()
    D = cdist(coords, coords)
    n = len(res_df)
    G = nx.Graph()
    for i, row in res_df.iterrows():
        G.add_node(int(i), **row.to_dict())
    for i in range(n):
        d = D[i].copy()
        d[i] = float("inf")
        for j in range(n):
            if abs(j - i) < min_sep:
                d[j] = float("inf")
        nbrs = np.argsort(d)[:topk]
        for j in nbrs:
            w = math.exp(-(D[i, j] ** 2) / (2 * sigma ** 2))
            G.add_edge(int(i), int(j), dist=float(D[i, j]), w_contact=float(w))
    return G

def laplacian_pinv_weighted(G: nx.Graph, eps=1e-3) -> np.ndarray:
    n = G.number_of_nodes()
    W = np.zeros((n, n), dtype=float)
    for i, j, data in G.edges(data=True):
        wij = float(data.get("w_contact", 1.0))
        if not np.isfinite(wij):
            wij = 0.0
        W[i, j] = wij
        W[j, i] = wij
    d = W.sum(axis=1)
    L = np.diag(d) - W
    return linalg.pinvh(L + eps * np.eye(n))

def prs_edge_flux(G: nx.Graph, Lp: np.ndarray, sources: List[int], sinks: List[int]) -> dict:
    n = G.number_of_nodes()
    b = np.zeros((n,))
    b[list(sources)] = 1.0
    b[list(sinks)] = -1.0
    phi = Lp @ b
    flux = {}
    for i, j, data in G.edges(data=True):
        R = data.get("dist", 1.0)
        R = float(R) if np.isfinite(R) else 1.0
        flux[(i, j)] = float(abs(phi[i] - phi[j]) / (R + 1e-6))
    return flux

def adapt_conductance(G: nx.Graph, flux: dict, alpha=0.2, key="g_mem", base=1.0) -> nx.Graph:
    for (i, j), f in flux.items():
        g = G.edges[i, j].get(key, base)
        if not np.isfinite(g):
            g = base
        G.edges[i, j][key] = (1 - alpha) * g + alpha * (base + float(f))
    return G

def _onehot_aa(aa1: str) -> np.ndarray:
    letters = "ACDEFGHIKLMNPQRSTVWY"
    v = np.zeros((20,), dtype=float)
    if aa1 in letters:
        v[letters.index(aa1)] = 1.0
    return v

def graph_features(G: nx.Graph):
    """Return (X, edges, E_np) with:
       X: [N, 3+20] (coords + one-hot AA)
       edges: list of (i,j,etype)
       E: [E, 3] -> [dist, w_contact, g_mem]
    """
    N = G.number_of_nodes()
    X = np.zeros((N, 3 + 20), dtype=np.float32)
    for i, data in G.nodes(data=True):
        X[i, 0:3] = np.array([data["x"], data["y"], data["z"]], dtype=np.float32)
        aa1 = AA3_TO1.get(data.get("resname", "GLY"), "G")
        X[i, 3:] = _onehot_aa(aa1)
    edges = []
    E = []
    for i, j, data in G.edges(data=True):
        gmem = float(data.get("g_mem", 1.0))
        E.append([float(data.get("dist", 1.0)), float(data.get("w_contact", 1.0)), gmem])
        edges.append((int(i), int(j), 0))
    return X, edges, np.array(E, dtype=np.float32)

def normalize_features_keep_gmem(X_np, E_np, node_mean, node_std, edge_mean, edge_std):
    # z-score nodes fully
    Xn = (X_np - node_mean) / (node_std + 1e-6)
    # z-score edges, but keep raw g_mem in column 2
    En = (E_np - edge_mean) / (edge_std + 1e-6)
    if E_np.shape[1] >= 3:
        En[:, 2] = E_np[:, 2]
    return Xn.astype(np.float32), En.astype(np.float32)

def to_features_device(G, scalers=None, device="cpu", requires_grad: bool=False):
    """
    Returns (X, edges, E) on device.
    - X is NOT grad-tracked (no need).
    - E is grad-tracked if requires_grad=True (needed for alignment/attribution losses).
    """
    import torch
    X_np, edges, E_np = graph_features(G)

    if scalers is not None:
        X_np, E_np = normalize_features_keep_gmem(
            X_np, E_np,
            scalers["node_mean"], scalers["node_std"],
            scalers["edge_mean"], scalers["edge_std"],
        )

    X = torch.tensor(X_np, dtype=torch.float32, device=device, requires_grad=False)
    # IMPORTANT: E must be a LEAF with requires_grad=True when requested
    E = torch.tensor(E_np, dtype=torch.float32, device=device, requires_grad=requires_grad)
    return X, edges, E

