#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train "memory" in a residue interaction network using the PRL-129-028101 rule.

Inputs
------
- A priors NPZ from rescontact (keys: priors [L,L,B], bins, mask, meta JSON)
- Optional PDB to provide CA-CA distances for edge lengths

Outputs
-------
- <outdir>/<query_id>_memory_edges.npz
  { edges (E,2), lengths (E,), C (E,), L, meta (json str) }
- <outdir>/<query_id>_power_vs_theta.csv
  columns: theta_deg, power

Usage (examples)
----------------
python packages/resintnet/scripts/train_memory.py \
  --priors data/templates/priors/106M_A.npz \
  --outdir outputs/memory \
  --edge-mode topk --topk 6 \
  --theta-train-deg 40 --iters 250

python packages/resintnet/scripts/train_memory.py \
  --priors data/templates/priors/106M_A.npz \
  --pdb data/pdb/train/106M.pdb --chain A \
  --edge-mode threshold --contact-thresh 8.0 --min-prob 0.15

Notes
-----
- Edge set comes from contact priors: either top-K per residue or prob-threshold.
- Edge lengths default to prior-expected distances; if --pdb is given, uses CA-CA.
- The conductance update enforces the "material constraint" each step.
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

# Optional: use Bio.PDB for CA coordinates if PDB provided
try:
    from Bio.PDB import PDBParser
except Exception:
    PDBParser = None

# Our PRL-faithful trainer
try:
    from resintnet.memory_flow import train_memory_network
except Exception as exc:
    print("[ERROR] Could not import resintnet.memory_flow. "
          "Make sure packages/resintnet/src is on PYTHONPATH.", file=sys.stderr)
    raise

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_priors(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    priors = z["priors"]           # (L, L, B)
    bins = z["bins"]               # bin edges (B+1,) or mids (B,)
    mask = z.get("mask", None)     # (L, L) or None
    meta = z.get("meta", None)
    if meta is None:
        meta = {"query_id": npz_path.stem.replace(".npz","")}
    else:
        try:
            meta = json.loads(str(meta))
        except Exception:
            meta = {"meta_raw": str(meta)}
    return priors, bins, mask, meta

def bin_midpoints(bins_arr: np.ndarray, B: int) -> np.ndarray:
    # If bins length equals B, assume already mids; else compute mids from edges
    if bins_arr.ndim == 1 and bins_arr.shape[0] == B:
        return bins_arr.astype(float)
    if bins_arr.ndim == 1 and bins_arr.shape[0] == B + 1:
        edges = bins_arr.astype(float)
        return 0.5 * (edges[:-1] + edges[1:])
    raise ValueError("Unrecognized bins format; expected (B,) or (B+1,)")

def expected_distances(priors: np.ndarray, mids: np.ndarray) -> np.ndarray:
    # E[d] = sum_b p_b * mid_b
    L, _, B = priors.shape
    mids_row = mids.reshape(1, 1, B)
    ed = np.sum(priors * mids_row, axis=2)
    # avoid zeros
    return np.clip(ed, 1e-6, None)

def contact_probability(priors: np.ndarray, mids: np.ndarray, thresh: float) -> np.ndarray:
    # Sum prob mass where bin midpoint <= thresh
    sel = mids <= float(thresh)
    pc = np.sum(priors[:, :, sel], axis=2)
    return pc

def ca_ca_distance_matrix(pdb_path: Path, chain_id: str | None) -> np.ndarray:
    if PDBParser is None:
        raise RuntimeError("Bio.PDB not available; install biopython to use --pdb.")
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", str(pdb_path))
    # Choose first model
    model = next(struct.get_models())
    if chain_id is None:
        # take first chain
        chain = next(model.get_chains())
    else:
        chain = model[chain_id]
    # Extract CA coordinates in residue order
    coords = []
    for res in chain:
        if "CA" in res:
            coords.append(res["CA"].coord.astype(float))
    if not coords:
        raise RuntimeError("No CA atoms found for the selected chain")
    X = np.stack(coords, axis=0)
    # pairwise distances
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    return D

def build_graph_from_priors(npz_path: Path,
                            edge_mode: str = "topk",
                            topk: int = 6,
                            min_prob: float = 0.2,
                            contact_thresh: float = 8.0,
                            pdb_path: Path | None = None,
                            pdb_chain: str | None = None):
    priors, bins, mask, meta = load_priors(npz_path)
    L, L2, B = priors.shape
    assert L == L2, "priors must be LxLxB"
    mids = bin_midpoints(bins, B)

    # base metrics
    Pcontact = contact_probability(priors, mids, contact_thresh)   # (L,L)
    Edist = expected_distances(priors, mids)                       # (L,L)

    # Optional: override lengths with CA-CA from PDB
    D_ca = None
    if pdb_path is not None:
        try:
            D_ca = ca_ca_distance_matrix(pdb_path, pdb_chain)
            if D_ca.shape[0] != L:
                print(f"[warn] CA-CA size {D_ca.shape[0]} != L={L}. "
                      "Falling back to expected distances for lengths.")
                D_ca = None
        except Exception as e:
            print(f"[warn] Failed to parse PDB for CA-CA: {e}. "
                  "Using expected distances.", file=sys.stderr)
            D_ca = None

    # Decide edges
    edges_set = set()
    if edge_mode == "topk":
        for i in range(L):
            probs = Pcontact[i].copy()
            probs[i] = -1.0
            if mask is not None:
                # mask False -> disallow
                probs[~mask[i].astype(bool)] = -1.0
            # pick topk indices
            js = np.argpartition(-probs, kth=min(topk, L-1)-1)[:topk]
            for j in js:
                if j == i: 
                    continue
                a, b = (i, j) if i < j else (j, i)
                edges_set.add((a, b))
    elif edge_mode == "threshold":
        sel = (Pcontact >= float(min_prob))
        if mask is not None:
            sel &= mask.astype(bool)
        idx = np.argwhere(sel)
        for i, j in idx:
            if i < j:
                edges_set.add((int(i), int(j)))
    else:
        raise ValueError("--edge-mode must be 'topk' or 'threshold'")

    if not edges_set:
        raise RuntimeError("No edges selected; relax --min-prob or increase --topk")

    edges = sorted(list(edges_set))
    E = len(edges)

    # Edge lengths and initial conductances
    lengths = np.zeros(E, dtype=float)
    for e_idx, (i, j) in enumerate(edges):
        if D_ca is not None:
            lengths[e_idx] = max(1e-6, float(D_ca[i, j]))
        else:
            lengths[e_idx] = max(1e-6, float(Edist[i, j]))

    C0 = 1e-2 * np.ones(E, dtype=float)

    meta_out = dict(meta)
    meta_out.update({
        "L": int(L),
        "edge_mode": edge_mode,
        "topk": int(topk),
        "min_prob": float(min_prob),
        "contact_thresh": float(contact_thresh),
        "length_source": "CA-CA" if D_ca is not None else "expected_distance",
        "npz_path": str(npz_path)
    })
    if pdb_path is not None:
        meta_out["pdb"] = str(pdb_path)
    if pdb_chain is not None:
        meta_out["pdb_chain"] = pdb_chain

    return L, np.array(edges, dtype=int), lengths, C0, meta_out

def main():
    ap = argparse.ArgumentParser(description="Train memory on a residue graph from rescontact priors.")
    ap.add_argument("--priors", required=True, type=Path, help="Path to rescontact priors NPZ")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--pdb", type=Path, default=None, help="Optional PDB path for CA-CA lengths")
    ap.add_argument("--chain", type=str, default=None, help="Chain ID in the PDB (e.g., 'A')")
    ap.add_argument("--edge-mode", choices=["topk", "threshold"], default="topk")
    ap.add_argument("--topk", type=int, default=6, help="Top-K neighbors per residue (edge-mode=topk)")
    ap.add_argument("--min-prob", type=float, default=0.20, help="Min contact prob (edge-mode=threshold)")
    ap.add_argument("--contact-thresh", type=float, default=8.0, help="Contact threshold in Å for P(contact)")
    ap.add_argument("--gamma", type=float, default=0.5, help="Material exponent γ (paper uses 0.5 often)")
    ap.add_argument("--K", type=float, default=1.0, help="Material budget K")
    ap.add_argument("--iters", type=int, default=200, help="Training iterations")
    ap.add_argument("--t-avg", type=int, default=16, help="Temporal averaging window for <Q^2>")
    ap.add_argument("--theta-train-deg", type=float, default=0.0, help="Training load direction (degrees)")
    ap.add_argument("--theta-grid", type=int, default=73, help="Number of angles to probe in [-180,180]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outlet", type=int, default=0, help="Pinned node index for Laplacian solve")
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    # Build graph
    L, edges, lengths, C0, meta = build_graph_from_priors(
        args.priors, edge_mode=args.edge_mode, topk=args.topk, min_prob=args.min_prob,
        contact_thresh=args.contact_thresh, pdb_path=args.pdb, pdb_chain=args.chain
    )

    # Train memory (PRL discrete update)
    from math import pi
    theta_train = args.theta_train_deg * pi / 180.0
    C_trained, probe = train_memory_network(
        n=L, edges=edges, Llen=lengths, C0=C0,
        gamma=args.gamma, K=args.K,
        theta_train=theta_train, T_avg=args.t_avg,
        iters=args.iters, outlet=args.outlet, rng=args.seed
    )

    # Probe power vs angle
    thetas = np.linspace(-np.pi, np.pi, num=args.theta_grid)
    Evals = probe(thetas)

    # Derive a sensible base name
    base = Path(args.priors).stem.replace(".npz", "")
    qid = meta.get("query_id", base)
    stem = f"{qid}".replace("/", "_")

    # Save NPZ of edges/lengths/conductances
    npz_out = args.outdir / f"{stem}_memory_edges.npz"
    meta_json = json.dumps(meta, ensure_ascii=False)
    np.savez_compressed(
        npz_out,
        edges=edges.astype(np.int32),
        lengths=lengths.astype(np.float32),
        C=C_trained.astype(np.float32),
        L=np.array([L], dtype=np.int32),
        meta=np.array(meta_json)
    )
    print(f"[memory] wrote {npz_out}")

    # Save CSV of power vs theta
    csv_out = args.outdir / f"{stem}_power_vs_theta.csv"
    degs = thetas * 180.0 / np.pi
    with open(csv_out, "w") as f:
        f.write("theta_deg,power\n")
        for d, e in zip(degs, Evals):
            f.write(f"{d:.6f},{e:.9e}\n")
    print(f"[memory] wrote {csv_out}")

    # Quick hint: the power should be minimal near theta_train_deg if memory formed
    print(f"[memory] train angle (deg) = {args.theta_train_deg:.2f}. "
          f"Min probe power at ~{float(degs[np.argmin(Evals)]):.2f}°")

if __name__ == "__main__":
    main()

