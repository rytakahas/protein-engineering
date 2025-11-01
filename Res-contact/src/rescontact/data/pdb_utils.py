# src/rescontact/data/pdb_utils.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import numpy as np

from Bio.PDB import PDBParser, MMCIFParser, PPBuilder

# ---------------------------------------------------------------------
# Notes (ground truth convention)
# ---------------------------------------------------------------------
# This implementation defines residue–residue contacts as:
#   Cα–Cα distance ≤ thr (Å), with pairs missing coordinates excluded.
#
# Extract Cα coordinates per residue and compute a symmetric L×L
# binary map Y with a valid-pair mask M. Residues lacking Cα are
# dropped during chain extraction to keep sequence/coords aligned.
#
# (Optional: Cβ–Cβ with Gly→Cα fallback, switch atom="CB" in
# _chain_seq_and_atom_coords (wrapper _chain_seq_and_ca keeps CA).)
# ---------------------------------------------------------------------


# --------------------------
# I/O + parsing helpers
# --------------------------
def load_structure(path: str):
    p = Path(path)
    if p.suffix.lower() in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(PERMISSIVE=True, QUIET=True)
    # structure id: filename stem
    return parser.get_structure(p.stem, str(p))


def _atom_coord_from_res(res, atom: str = "CA") -> Optional[np.ndarray]:
    """
    Return 3D coordinate for the requested atom in a Biopython Residue.
    If atom == "CB" and residue is GLY, fall back to "CA" (common convention).
    Returns None if the atom is missing.
    """
    atom_name = atom
    try:
        if atom.upper() == "CB":
            # Glycine has no CB; use CA as standard fallback
            resname = res.get_resname().strip().upper()
            if resname in ("GLY", "GLY A", "GLY  "):
                atom_name = "CA"
        if atom_name in res:
            v = res[atom_name].get_vector()
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
    except Exception:
        pass
    return None


def _chain_seq_and_atom_coords(
    chain,
    atom: str = "CA",
) -> Tuple[str, np.ndarray]:
    """
    Returns (sequence_str, coords[L,3]) for a chain using PPBuilder peptides,
    aligned so the i-th character of sequence matches coords[i].
    Residues missing the requested atom are dropped to keep alignment.

    atom: "CA" (default) or "CB" (with Gly→CA fallback handled internally).
    """
    ppb = PPBuilder()
    seq_parts: List[str] = []
    coords_parts: List[np.ndarray] = []

    for poly in ppb.build_peptides(chain):
        seq = str(poly.get_sequence())  # includes all residues in the peptide
        kept_coords: List[np.ndarray] = []
        kept_chars: List[str] = []

        # Iterate residues in the same order PPBuilder used for the sequence
        for idx, res in enumerate(poly):
            coord = _atom_coord_from_res(res, atom=atom)
            if coord is None:
                # Skip this residue (both from sequence and coords) to keep L aligned
                continue
            # Keep one character from the peptide sequence for alignment
            # (PolyResidue order matches seq string produced by PPBuilder)
            if idx < len(seq):
                kept_chars.append(seq[idx])
            else:
                # Fallback, normally not hit
                kept_chars.append("X")
            kept_coords.append(coord)

        if kept_coords and kept_chars:
            seq_parts.append("".join(kept_chars))
            coords_parts.append(np.asarray(kept_coords, dtype=np.float32))

    if not seq_parts:
        return "", np.zeros((0, 3), dtype=np.float32)

    seq_full = "".join(seq_parts)
    coords_full = np.concatenate(coords_parts, axis=0)
    L = min(len(seq_full), coords_full.shape[0])
    return seq_full[:L], coords_full[:L]


def _chain_seq_and_ca(chain) -> Tuple[str, np.ndarray]:
    """
    Backward-compatible wrapper that returns (sequence, CA coords[L,3]).
    Ground truth in this codebase is Cα–Cα ≤ threshold by default.
    """
    return _chain_seq_and_atom_coords(chain, atom="CA")


def _structure_chains(structure) -> Dict[str, Tuple[str, np.ndarray]]:
    """
    Map chain_id -> (sequence, ca_coords[L,3]), skipping empty chains.
    Uses Cα coordinates (ground-truth definition).
    """
    out: Dict[str, Tuple[str, np.ndarray]] = {}
    # take first model
    model = next(structure.get_models())
    for chain in model.get_chains():
        cid = chain.id
        seq, coords = _chain_seq_and_ca(chain)
        if len(seq) >= 2 and coords.shape[0] >= 2:
            out[cid] = (seq, coords)
    return out


# --------------------------
# Contact map (intra-chain)
# --------------------------
def contact_map_from_coords(coords: np.ndarray, thr: float, sym: bool = True):
    """
    coords: [L,3] (float32) Cα coordinates.
    thr   : contact threshold (Å), e.g., 8.0
    Returns:
      Y: [L,L] float32 {0,1} contact labels (diag=0)
      M: [L,L] float32 {0,1} valid-pair mask (1 where both residues have coords)
    """
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        L = int(coords.shape[0]) if coords.ndim >= 1 else 0
        return np.zeros((L, L), dtype=np.float32), np.zeros((L, L), dtype=np.float32)

    # mark rows that are finite
    row_ok = np.isfinite(coords).all(axis=1)
    valid = np.where(row_ok)[0]
    L = coords.shape[0]
    Y = np.zeros((L, L), dtype=np.float32)
    M = np.zeros((L, L), dtype=np.float32)
    if valid.size == 0:
        return Y, M

    c = coords[valid]  # [L_valid,3]
    # pairwise squared distances
    diff = c[:, None, :] - c[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    contact_bool = (d2 <= (thr * thr)).astype(np.float32)

    # scatter back into full map
    for i, ii in enumerate(valid):
        for j, jj in enumerate(valid):
            Y[ii, jj] = contact_bool[i, j]
            M[ii, jj] = 1.0

    if sym:
        Y = np.maximum(Y, Y.T)
        M = np.maximum(M, M.T)

    # zero diagonal
    np.fill_diagonal(Y, 0.0)
    return Y, M


# --------------------------
# Contact map (inter-chain; optional utility)
# --------------------------
def contact_map_between_coords(coords_a: np.ndarray, coords_b: np.ndarray, thr: float):
    """
    Rectangular contact map between chains A (L1,3) and B (L2,3), using Cα coords.
    Returns:
      Yab: [L1,L2] {0,1}
      Mab: [L1,L2] {0,1}
    """
    if coords_a.ndim != 2 or coords_b.ndim != 2 or coords_a.shape[1] != 3 or coords_b.shape[1] != 3:
        L1 = int(coords_a.shape[0]) if coords_a.ndim >= 1 else 0
        L2 = int(coords_b.shape[0]) if coords_b.ndim >= 1 else 0
        return np.zeros((L1, L2), dtype=np.float32), np.zeros((L1, L2), dtype=np.float32)

    ok_a = np.isfinite(coords_a).all(axis=1)
    ok_b = np.isfinite(coords_b).all(axis=1)
    idx_a = np.where(ok_a)[0]
    idx_b = np.where(ok_b)[0]

    L1, L2 = coords_a.shape[0], coords_b.shape[0]
    Y = np.zeros((L1, L2), dtype=np.float32)
    M = np.zeros((L1, L2), dtype=np.float32)
    if idx_a.size == 0 or idx_b.size == 0:
        return Y, M

    ca = coords_a[idx_a]
    cb = coords_b[idx_b]
    diff = ca[:, None, :] - cb[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    contact_bool = (d2 <= (thr * thr)).astype(np.float32)

    for ia, aa in enumerate(idx_a):
        for ib, bb in enumerate(idx_b):
            Y[aa, bb] = contact_bool[ia, ib]
            M[aa, bb] = 1.0

    return Y, M


# --------------------------
# Enumeration
# --------------------------
def enumerate_examples_from_dir(
    root_dir: str,
    include_inter_chain: bool,
    limit: Optional[int] = None,
    list_first_n: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Yields examples with keys:
      - example_id: "file.pdb::A" (intra) or "file.pdb::A+B" (inter)
      - sequences: [seqA] (intra) or [seqA, seqB] (inter)
      - coords:    [coordsA] or [coordsA, coordsB]  (Cα coords)
      - kind:      "intra" or "inter"
    Stops once `limit` examples have been yielded.

    For inter-chain pairs, you can build rectangular contact maps with
    contact_map_between_coords(coordsA, coordsB, thr).
    """
    root = Path(root_dir)
    files = sorted(
        [p for p in root.rglob("*") if p.suffix.lower() in (".pdb", ".ent", ".cif", ".mmcif")]
    )
    if list_first_n and list_first_n > 0:
        print(f"[rescontact/ds] pre-scan: found {len(files)} structure files under {root_dir}")
        for p in files[: list_first_n]:
            print("  -", p.relative_to(root))

    produced = 0
    for fp in files:
        if limit is not None and produced >= limit:
            break
        try:
            s = load_structure(str(fp))
            chains = _structure_chains(s)  # dict cid -> (seq, coords[L,3]) using Cα
        except Exception:
            continue
        if not chains:
            continue

        # Intra per chain
        for cid, (seq, coords) in chains.items():
            if limit is not None and produced >= limit:
                break
            if len(seq) < 2 or coords.shape[0] < 2:
                continue
            yield {
                "example_id": f"{fp.name}::{cid}",
                "sequences": [seq],
                "coords": [coords],
                "kind": "intra",
            }
            produced += 1

        if limit is not None and produced >= limit:
            break

        # Inter-chain (unordered pairs A<B)
        if include_inter_chain and len(chains) >= 2:
            cids = sorted(chains.keys())
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    if limit is not None and produced >= limit:
                        break
                    ci, cj = cids[i], cids[j]
                    seq_i, coords_i = chains[ci]
                    seq_j, coords_j = chains[cj]
                    if len(seq_i) < 2 or len(seq_j) < 2:
                        continue
                    yield {
                        "example_id": f"{fp.name}::{ci}+{cj}",
                        "sequences": [seq_i, seq_j],
                        "coords": [coords_i, coords_j],
                        "kind": "inter",
                    }
                    produced += 1

        if limit is not None and produced >= limit:
            break
