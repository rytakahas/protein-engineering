# src/resintnet/memory_flow.py
# -*- coding: utf-8 -*-
import os, io, math, json, numpy as np, pandas as pd
from typing import List, Tuple, Optional
from tqdm.auto import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch

from .graph import (
    parse_cif_to_residues, build_topk_graph, laplacian_pinv_weighted,
    prs_edge_flux, adapt_conductance, graph_features, to_features_device,
    normalize_features_keep_gmem
)
from .models.sage import GraphBatchNetAMP
from .ingest import get_mavedb_first_scores_for, get_fireprot_labels_for

# ---------------------------
# HTTP helpers
# ---------------------------
def make_session(total=3, backoff=0.5, status=(500,502,503,504)):
    s = requests.Session()
    r = Retry(total=total, read=total, connect=total, backoff_factor=backoff, status_forcelist=status)
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": "resintnet/0.1"})
    return s
SESSION = make_session()

BASES = {
    "UNIPROT_SEARCH": "https://rest.uniprot.org/uniprotkb/search",
    "PDBe_BEST":      "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures",
    "RCSB_FILES":     "https://files.rcsb.org/download",
    "AFDB_FILE":      "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.cif",
}

# ---------------------------
# Data collection
# ---------------------------
def collect_uniprot_via_terms(terms: List[str], size: int=5) -> List[str]:
    accs = []
    for t in terms:
        r = SESSION.get(BASES["UNIPROT_SEARCH"], params={"query":t, "format":"tsv", "fields":"accession,id", "size":500}, timeout=60)
        if r.status_code == 200:
            import io
            df = pd.read_csv(io.StringIO(r.text), sep="\t")
            accs.extend(list(df["Entry"][:size]))
    seen=set(); out=[]
    for a in accs:
        if a not in seen:
            out.append(a); seen.add(a)
    return out

def pdbe_best_structures(acc: str):
    r = SESSION.get(f"{BASES['PDBe_BEST']}/{acc}", timeout=60)
    if r.status_code != 200: return []
    return r.json().get(acc, [])

def download_pdb_cif(pdb_id: str, out_dir) -> str:
    out = os.path.join(out_dir, f"{pdb_id.upper()}.cif")
    if not os.path.exists(out):
        r = SESSION.get(f"{BASES['RCSB_FILES']}/{pdb_id.upper()}.cif", timeout=180)
        r.raise_for_status()
        open(out, "w").write(r.text)
    return out

def download_afdb_cif(acc: str, out_dir) -> Optional[str]:
    out = os.path.join(out_dir, f"AF-{acc}-F1-model_v4.cif")
    if not os.path.exists(out):
        r = SESSION.get(BASES["AFDB_FILE"].format(acc=acc), timeout=180)
        if r.status_code != 200: return None
        open(out, "w").write(r.text)
    return out

def uniprot_to_structures(accs: List[str], cap: int=100) -> pd.DataFrame:
    rows=[]
    for acc in accs[:cap]:
        best = pdbe_best_structures(acc)
        if best:
            def score_row(r):
                cov = float(r.get("coverage", 0.0) or 0.0)
                res = r.get("resolution", 9.9)
                try: res = float(res) if res is not None else 9.9
                except: res = 9.9
                return (cov, -res)
            top = max(best, key=score_row)
            rows.append({"uniprot":acc,"pdb_id":str(top.get("pdb_id","")).upper(),
                         "chain_id":top.get("chain_id",""),"coverage":top.get("coverage"),
                         "resolution":top.get("resolution"),"source":"PDB"})
        else:
            rows.append({"uniprot":acc,"pdb_id":None,"chain_id":"",
                         "coverage":None,"resolution":None,"source":"AFDB"})
    return pd.DataFrame(rows)

def build_graphs_from_structures(sel_df: pd.DataFrame, pdb_cache: str, afdb_cache: str,
                                 topk=6, min_sep=2, sigma=8.0, lap_eps=1e-3):
    os.makedirs(pdb_cache, exist_ok=True)
    os.makedirs(afdb_cache, exist_ok=True)
    paths=[]
    for _,row in tqdm(sel_df.iterrows(), total=len(sel_df), desc="Downloading structures"):
        try:
            if row["source"]=="PDB" and row["pdb_id"]:
                paths.append(download_pdb_cif(row["pdb_id"], pdb_cache))
            else:
                af=download_afdb_cif(row["uniprot"], afdb_cache)
                paths.append(af)
        except Exception:
            paths.append(None)
    sel = sel_df.copy(); sel["cif_path"]=paths
    sel = sel.dropna(subset=["cif_path"]).reset_index(drop=True)

    graphs=[]
    for _,row in tqdm(sel.iterrows(), total=len(sel), desc="Building graphs"):
        try:
            res = parse_cif_to_residues(row["cif_path"], chain_id=row.get("chain_id") or None)
            if len(res) < 8:
                continue
            G = build_topk_graph(res, topk=topk, min_sep=min_sep, sigma=sigma)
            Lp = laplacian_pinv_weighted(G, eps=lap_eps)
            flux = prs_edge_flux(G, Lp, [0], [min(6, len(res)-1)])
            adapt_conductance(G, flux, alpha=0.3, key="g_mem", base=1.0)
            # scale g_mem to ~[0,1] to keep gate well-conditioned
            gvals = [G.edges[i,j]["g_mem"] for i,j in G.edges()]
            if len(gvals)>0:
                gmin, gmax = float(np.min(gvals)), float(np.max(gvals))
                for (i,j) in G.edges():
                    g = G.edges[i,j]["g_mem"]
                    G.edges[i,j]["g_mem"] = 0.0 if gmax==gmin else (g - gmin)/(gmax - gmin + 1e-12)
            graphs.append((row["uniprot"], row.get("pdb_id"), row.get("chain_id"), G))
        except Exception as e:
            print("Graph build failed:", row.to_dict(), e)
    return graphs

# ---------------------------
# Targets (weak graph-level)
# ---------------------------
def _finite(x, default=0.0):
    return float(x) if np.isfinite(x) else float(default)

def graph_targets_for_uniprot(acc: str):
    A_t = 0.0; ddG_t = 0.0
    dms_df = get_mavedb_first_scores_for(acc)
    if dms_df is not None and "score" in dms_df:
        m = pd.to_numeric(dms_df["score"], errors="coerce").dropna()
        A_t = _finite(m.mean(), 0.0) if len(m) else 0.0
    fp_df = get_fireprot_labels_for(acc)
    if fp_df is not None:
        lc = {c.lower(): c for c in fp_df.columns}
        for k in ["ddg","delta_delta_g","delta g","ΔΔg","delta_g","ddg_pred","ddg_experimental"]:
            if k in lc:
                m = pd.to_numeric(fp_df[lc[k]], errors="coerce").dropna()
                ddG_t = _finite(m.mean(), 0.0) if len(m) else 0.0
                break
    return A_t, ddG_t

# ---------------------------
# Training (route + alignment)
# ---------------------------
def fit_feature_scalers(graphs):
    Xs, Es = [], []
    for (_,_,_,G) in graphs:
        X_np, edges, E_np = graph_features(G)
        Xs.append(X_np); Es.append(E_np)
    Xcat = np.concatenate(Xs, axis=0)
    Ecat = np.concatenate(Es, axis=0) if len(Es) else np.zeros((1,3), dtype=float)
    return {
        "node_mean": Xcat.mean(axis=0, keepdims=True),
        "node_std":  Xcat.std(axis=0, keepdims=True) + 1e-6,
        "edge_mean": Ecat.mean(axis=0, keepdims=True),
        "edge_std":  Ecat.std(axis=0, keepdims=True) + 1e-6,
    }

def train_graph_regressor_amp_route(graphs, scalers, steps=300, lr=1e-3, weight_decay=1e-4, clip=1.0,
                                    route_w=0.3, align_w=0.3):
    """
    Training with:
      - main MSE on weak graph-level targets (A_t, ddG_t),
      - route supervision: gate(E[:,2]) ~ normalized PRS flux,
      - gradient-alignment: || ∂A/∂g_mem || (normalized) ~ PRS flux.
    IMPORTANT: tensors used for alignment must require_grad=True.
    """
    X0,_,E0 = to_features_device(graphs[0][3], scalers, device="cpu")
    model = GraphBatchNetAMP(X0.shape[1], E0.shape[1]).to("cpu")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    huber = torch.nn.SmoothL1Loss()

    # cache PRS-aligned flux vectors (per-graph, aligned with edge order)
    cache = []
    for (acc, pdbid, chain, G) in graphs:
        X0, edges, E0 = to_features_device(G, scalers, device="cpu")
        Lp = laplacian_pinv_weighted(G, eps=1e-3)
        n = G.number_of_nodes()
        flux_map = prs_edge_flux(G, Lp, [0], [min(6, n-1)])
        f = np.zeros((len(edges),), dtype=float)
        for k,(i,j,_) in enumerate(edges):
            f[k] = float(flux_map.get((i,j), flux_map.get((j,i), 0.0)))
        if np.max(f)>0:
            f = f / (np.max(f) + 1e-12)
        f = 0.05 + 0.95*f
        cache.append((acc, pdbid, chain, G, edges, f))

    pbar = tqdm(range(steps), desc="Training(route)")
    for step in pbar:
        model.train(); opt.zero_grad(set_to_none=True)
        idx = np.random.choice(len(cache), min(4, len(cache)), replace=False)

        feats, tgtY, route_targets = [], [], []
        for ii in idx:
            acc, pdbid, chain, G, edges, f = cache[ii]
            # requires_grad=True to enable gradient-alignment on E later
            X, edges, E = to_features_device(G, scalers, device="cpu", requires_grad=True)
            feats.append((X, edges, E))
            A_t, ddG_t = graph_targets_for_uniprot(acc)
            tgtY.append([A_t, ddG_t])
            route_targets.append(torch.tensor(f, dtype=torch.float32))

        tgt = torch.tensor(tgtY, dtype=torch.float32)
        with torch.no_grad():
            tgt = torch.nan_to_num(tgt, nan=0.0, posinf=0.0, neginf=0.0)

        y = model(feats)
        y = torch.nan_to_num(y, 0.0, 0.0, 0.0)
        loss_main = torch.nn.functional.mse_loss(y, tgt)

        # route + alignment losses
        route_loss = 0.0
        align_loss = 0.0
        for (X,edges,E), flux_t in zip(feats, route_targets):
            if E.shape[1] >= 3:
                # route: make gate(E[:,2]) follow PRS flux
                g = E[:,2]
                # mimic gate used in the model (softplus on centered g)
                # NOTE: use model params to keep shapes consistent
                gate = torch.nn.functional.softplus(model.gate_scale * (g - model.g_gate_center)) + 1e-3
                gate_n = (gate - gate.min()) / (gate.max() - gate.min() + 1e-6)
                route_loss = route_loss + huber(gate_n, flux_t.to(gate_n.dtype))

                # alignment: match saliency wrt g_mem to PRS flux
                Ah,_ = model([(X,edges,E)])[0]
                grads = torch.autograd.grad(Ah, E, retain_graph=True, create_graph=True)[0][:,2]
                g_norm = grads.abs() / (grads.abs().max() + 1e-6)
                align_loss = align_loss + huber(g_norm, flux_t.to(g_norm.dtype))

        loss = loss_main + route_w * route_loss + align_w * align_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        opt.step()

        pbar.set_postfix({
            "loss": float(loss.detach().cpu()),
            "route": float((route_w * (route_loss if isinstance(route_loss, torch.Tensor) else 0)).detach().cpu()) if isinstance(route_loss, torch.Tensor) else 0.0,
            "align": float((align_w * (align_loss if isinstance(align_loss, torch.Tensor) else 0)).detach().cpu()) if isinstance(align_loss, torch.Tensor) else 0.0,
        })
    return model

# ---------------------------
# Inference / utils
# ---------------------------
def predict_graph_amp(model, G, scalers=None):
    X, edges, E = to_features_device(G, scalers, device="cpu")
    with torch.no_grad():
        A_hat, ddG_hat = model([(X,edges,E)])[0]
    return float(A_hat.detach().cpu()), float(ddG_hat.detach().cpu())

def clone_graph(G):
    import networkx as nx
    H = nx.Graph()
    H.add_nodes_from((i, data.copy()) for i, data in G.nodes(data=True))
    H.add_edges_from((i, j, data.copy()) for i, j, data in G.edges(data=True))
    return H

def set_node_aa(G, i, aa1):
    from .graph import AA1_TO3
    aa1 = str(aa1).upper()
    res3 = AA1_TO3.get(aa1, "GLY")
    G.nodes[i]["resname"] = res3

def seq_from_graph(G):
    from .graph import AA3_TO1
    s = []
    for i,_ in G.nodes(data=True):
        aa1 = AA3_TO1.get(G.nodes[i].get("resname","GLY"), "G")
        s.append(aa1)
    return "".join(s)

def mutational_scan_amp(model, G, scalers=None, uni="UNK", pdbid=None, chain=None, reward_w=(1.0,0.5,0.05), lap_eps=1e-3):
    base_seq = seq_from_graph(G)
    A0, ddG0 = predict_graph_amp(model, G, scalers)
    rows = []
    n = G.number_of_nodes()
    letters = "ACDEFGHIKLMNPQRSTVWY"
    for i in tqdm(range(n), desc="Scan positions"):
        wt = base_seq[i]
        for aa in letters:
            if aa == wt: continue
            H = clone_graph(G); set_node_aa(H, i, aa)
            Lp = laplacian_pinv_weighted(H, eps=lap_eps)
            flux = prs_edge_flux(H, Lp, [i], [min(i+5, n-1)])
            adapt_conductance(H, flux, alpha=0.05, key="g_mem", base=1.0)
            A1, ddG1 = predict_graph_amp(model, H, scalers)
            dA = A1 - A0
            ddG_pos = max(0.0, ddG1)
            reward = reward_w[0]*dA - reward_w[1]*ddG_pos - reward_w[2]*1.0
            rows.append({
                "uniprot": uni, "pdb_id": pdbid, "chain": chain,
                "pos": i, "wt": wt, "mut": aa,
                "A_hat": A1, "ddG_hat": ddG1, "delta_A": dA,
                "stability_penalty": ddG_pos, "reward": reward,
                "seq_before": base_seq, "seq_after": base_seq[:i]+aa+base_seq[i+1:]
            })
    df = pd.DataFrame(rows).sort_values("reward", ascending=False).reset_index(drop=True)
    return df

def influence_scores_amp(model, G, scalers=None, out_dir=None, tag=""):
    import numpy as np
    X, edges, E = to_features_device(G, scalers, device="cpu", requires_grad=True)
    if hasattr(E, "retain_grad"): E.retain_grad()
    A_hat, ddG_hat = model([(X, edges, E)])[0]
    model.zero_grad(set_to_none=True)
    A_hat.backward(retain_graph=True)
    gradX_A = X.grad.detach().cpu().numpy()
    gradE_A = E.grad.detach().cpu().numpy()
    model.zero_grad(set_to_none=True); X.grad = None; E.grad = None
    ddG_hat.backward()
    gradX_D = X.grad.detach().cpu().numpy()
    gradE_D = E.grad.detach().cpu().numpy()

    node_inf = pd.DataFrame({
        "node": np.arange(X.shape[0]),
        "grad_norm_A": np.linalg.norm(gradX_A, axis=1),
        "grad_norm_ddG": np.linalg.norm(gradX_D, axis=1),
    })
    edge_rows = []
    for k,(i,j,_) in enumerate(edges):
        gA = float(abs(gradE_A[k,2])) if E.shape[1] >= 3 else float("nan")
        gD = float(abs(gradE_D[k,2])) if E.shape[1] >= 3 else float("nan")
        edge_rows.append({"i": i, "j": j, "grad_gmem_A": gA, "grad_gmem_ddG": gD})
    edge_inf = pd.DataFrame(edge_rows).sort_values("grad_gmem_A", ascending=False).reset_index(drop=True)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        node_p = os.path.join(out_dir, f"influence_nodes{tag}.csv")
        edge_p = os.path.join(out_dir, f"influence_edges{tag}.csv")
        node_inf.to_csv(node_p, index=False); edge_inf.to_csv(edge_p, index=False)
        return node_inf, edge_inf, node_p, edge_p
    return node_inf, edge_inf, None, None

def validate_influence_edges_sweep(model, G, scalers=None, eps_list=(0.1,0.25,0.5,1.0), sign_tau=1e-5):
    import numpy as np, pandas as pd
    X, edges, E = to_features_device(G, scalers, device="cpu", requires_grad=True)
    if hasattr(E, "retain_grad"): E.retain_grad()
    Ah, _ = model([(X,edges,E)])[0]
    Ah = float(Ah.detach().cpu())
    res=[]
    for eps in eps_list:
        X, edges, E = to_features_device(G, scalers, device="cpu", requires_grad=True)
        if hasattr(E, "retain_grad"): E.retain_grad()
        A_hat, _ = model([(X,edges,E)])[0]
        model.zero_grad(set_to_none=True); A_hat.backward(retain_graph=True)
        ggrad = E.grad[:,2].detach().cpu().numpy() if E.shape[1] >= 3 else np.zeros((len(edges),))
        order = np.argsort(-np.abs(ggrad))
        k_use = min(10, len(order))
        ks = list(order[:k_use]) + list(order[-k_use:])
        dA_pred, dA_meas = [], []
        for k in ks:
            i,j,_ = edges[k]
            H = clone_graph(G)
            old = H.edges[i,j].get("g_mem", 1.0)
            H.edges[i,j]["g_mem"] = float(old) + float(eps)
            A1,_ = predict_graph_amp(model, H, scalers)
            dA_meas.append(A1 - Ah)
            dA_pred.append(ggrad[k] * eps)
        dA_pred = np.array(dA_pred); dA_meas = np.array(dA_meas)
        mask = (np.abs(dA_pred)>sign_tau) & (np.abs(dA_meas)>sign_tau)
        sign_agree = float((np.sign(dA_pred[mask]) == np.sign(dA_meas[mask])).mean()) if mask.any() else np.nan
        rx = dA_pred.argsort().argsort(); ry = dA_meas.argsort().argsort()
        spear = float(np.corrcoef(rx, ry)[0,1]) if len(rx)>1 else np.nan
        xm, ym = dA_pred - dA_pred.mean(), dA_meas - dA_meas.mean()
        denom = (np.sqrt((xm**2).sum()*(ym**2).sum()) + 1e-12)
        pear = float((xm*ym).sum()/denom)
        res.append({"eps":eps, "pearson":pear, "spearman":spear, "sign_agree_masked":sign_agree,
                    "mean_abs_dA_meas": float(np.mean(np.abs(dA_meas)))})
    return pd.DataFrame(res)

def edge_grad_rank(model, G, scalers=None, head="A"):
    X, edges, E = to_features_device(G, scalers, device="cpu", requires_grad=True)
    if hasattr(E, "retain_grad"): E.retain_grad()
    A_hat, ddG_hat = model([(X,edges,E)])[0]
    model.zero_grad(set_to_none=True)
    (A_hat if head=="A" else ddG_hat).backward(retain_graph=True)
    g = E.grad[:,2].detach().abs().cpu().numpy() if E.shape[1] >= 3 else None
    if g is None: return []
    return g.argsort()[::-1]

def ablate_edges_by_rank(model, G, scalers=None, k=10, eps=0.5):
    A0,_ = predict_graph_amp(model, G, scalers)
    order = edge_grad_rank(model, G, scalers, head="A")
    if len(order)==0:
        return {"top": None, "bottom": None}
    XH, edgesH, EH = to_features_device(G, scalers, device="cpu")
    res={}
    for label, idxs in [("top", order[:k]), ("bottom", order[-k:])]:
        H = clone_graph(G)
        XH, edgesH, EH = to_features_device(H, scalers, device="cpu")
        for eidx in idxs:
            i,j,_ = edgesH[eidx]
            H.edges[i,j]["g_mem"] = H.edges[i,j].get("g_mem",0.0) + float(eps)
        A1,_ = predict_graph_amp(model, H, scalers)
        res[label] = float(A1 - A0)
    return res

# ---------------------------
# RL refinement (logit-space)
# ---------------------------
def rl_refine_gmem(model, G, scalers=None, steps=150, lr=1.2, lam_ddg=0.3, top_frac=0.3, momentum=0.8):
    """
    Projected gradient ascent on g_mem with:
      - grad normalization (per-graph)
      - edge importance weights (PRS flux)
      - momentum
      - stronger clamp range
    """
    import numpy as np, torch
    from .graph import laplacian_pinv_weighted, prs_edge_flux

    # Edge list + initial flux weights in [0.05, 1.0]
    X0, edges, E0 = to_features_device(G, scalers, device="cpu", requires_grad=False)
    Lp = laplacian_pinv_weighted(G, eps=1e-3)
    n = G.number_of_nodes()
    flux_map = prs_edge_flux(G, Lp, [0], [min(6, n-1)])
    w = np.zeros((len(edges),), dtype=float)
    for k,(i,j,_) in enumerate(edges):
        w[k] = float(flux_map.get((i,j), flux_map.get((j,i), 0.0)))
    if w.max() > 0:
        w = w / (w.max() + 1e-12)
    w = 0.05 + 0.95 * w  # avoid zeros
    v = np.zeros_like(w) # momentum buffer

    # Which edges we’re allowed to move most: top fraction by current saliency
    def _saliency_rank():
        X, e, E = to_features_device(G, scalers, device="cpu", requires_grad=True)
        A_hat, _ = model([(X,e,E)])[0]
        model.zero_grad(set_to_none=True)
        A_hat.backward()
        g = E.grad[:,2].detach().abs().cpu().numpy()
        order = g.argsort()[::-1]
        k = max(1, int(round(top_frac * len(order))))
        return set(order[:k]), g.mean()

    top_set, gbar = _saliency_rank()
    for t in range(steps):
        X, e, E = to_features_device(G, scalers, device="cpu", requires_grad=True)
        A_hat, ddG_hat = model([(X,e,E)])[0]
        R = A_hat - lam_ddg * torch.relu(ddG_hat)
        gradE = torch.autograd.grad(R, E, retain_graph=False)[0]
        g_raw = gradE[:,2].detach().cpu().numpy()

        # normalize gradient per-step
        g_norm = g_raw / (np.abs(g_raw).mean() + 1e-8)
        # emphasize PRS-important edges and top-saliency subset
        mask = np.array([1.0 if idx in top_set else 0.4 for idx in range(len(e))], dtype=float)
        step_vec = w * mask * g_norm

        # momentum + update
        v = momentum * v + (1.0 - momentum) * step_vec
        for k,(i,j,_) in enumerate(e):
            G.edges[i,j]["g_mem"] = float(G.edges[i,j].get("g_mem",1.0) + lr * v[k])
            # wider range so updates can actually matter
            G.edges[i,j]["g_mem"] = float(max(-0.5, min(5.0, G.edges[i,j]["g_mem"])))

        if t % 10 == 0 or t == steps - 1:
            # saturation report + quick ablation delta
            Xh, eh, Eh = to_features_device(G, scalers)
            gvals = Eh[:,2].detach().cpu().numpy()
            sat_lo = float((gvals <= -0.45).mean())
            sat_hi = float((gvals >= 4.9).mean())
            # small K to keep it cheap
            from .memory_flow import ablate_edges_by_rank
            delta = ablate_edges_by_rank(model, G, scalers, k=10, eps=0.5)
            print(f"[RL t={t}] mean|grad|={np.abs(g_raw).mean():.2e}  sat_lo={sat_lo:.2f}  sat_hi={sat_hi:.2f}  ΔA_top={delta.get('top')}  ΔA_bot={delta.get('bottom')}")

        # re-evaluate saliency every 20 steps
        if (t+1) % 20 == 0:
            top_set, gbar = _saliency_rank()

    return G


