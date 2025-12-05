# src/rescontact/templates/features.py
"""
Build template priors from a structure file (.pdb/.cif).

We compute C-alpha distance matrix for the template (in template index space),
then project it to query index space via the mapping.
We produce:
  - contact_prior (L, L) float32, values in [0,1] as a soft prior
  - dist_bin_logits (L, L, B) float32, where bins are distance thresholds (Å)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

# Minimal PDB parser for CA atoms
def _read_ca_coords(pdb_path: Path):
    coords = []  # (resi_index, x, y, z)
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            seen = set()
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                name = line[12:16].strip()
                if name != "CA":
                    continue
                resi = int(line[22:26].strip())
                # avoid duplicates
                key = resi
                if key in seen:
                    continue
                seen.add(key)
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                coords.append((resi, x, y, z))
    except Exception:
        return None
    if not coords:
        return None
    # Map residue indices to a dense array by rank order
    coords.sort(key=lambda t: t[0])
    xs = np.array([c[1] for c in coords], dtype=np.float32)
    ys = np.array([c[2] for c in coords], dtype=np.float32)
    zs = np.array([c[3] for c in coords], dtype=np.float32)
    return np.stack([xs, ys, zs], axis=1)  # (T, 3)

def priors_from_structure(struct_path: Path, mapping: List[Tuple[int,int]], L: int, bin_edges: List[float]):
    ca = _read_ca_coords(struct_path)
    if ca is None:
        return None
    T = ca.shape[0]
    # compute pairwise dists in template index space (dense order)
    # for simplicity, assume mapping uses template positions in [0..T-1] ranks (not PDB residue numbers)
    # If your mapping uses PDB residue numbers, you should pre-map them to [0..T-1].
    # Here we defensively clip indices.
    # Distance matrix
    diff = ca[:, None, :] - ca[None, :, :]
    D = np.sqrt((diff**2).sum(axis=-1))  # (T, T)

    # Project to query
    contact_prior = np.zeros((L, L), dtype=np.float32)
    B = len(bin_edges)
    dist_bin_logits = np.zeros((L, L, B), dtype=np.float32)

    for tpos, qpos in mapping:
        if tpos < 0 or tpos >= T or qpos < 0 or qpos >= L:
            continue
        # For each mapped pair (tpos, qpos), transfer a row/column slice
        # Simple approach: nearest-neighbor projection along mapped rows/cols
        # Row tpos -> row qpos  ;  Col tpos -> col qpos
        # This double-projection is crude but works as a soft prior.
        # Row projection
        trow = D[tpos, :]  # (T,)
        # Map the partner columns via mapping as well
        for tpos2, qpos2 in mapping:
            if tpos2 < 0 or tpos2 >= T or qpos2 < 0 or qpos2 >= L:
                continue
            d = float(trow[tpos2])
            # Contact soft prior: 1 if d<=8, else decays with distance
            contact_prior[qpos, qpos2] = max(contact_prior[qpos, qpos2], 1.0 / (1.0 + (d / 8.0)**2))
            # Distance-bin logits (+1 in the bin that matches d)
            for b, edge in enumerate(bin_edges):
                if d <= edge:
                    dist_bin_logits[qpos, qpos2, b] += 1.0
                    break

    return {
        "contact_prior": contact_prior,
        "dist_bin_logits": dist_bin_logits,
        "source": "structure_projected"
    }
