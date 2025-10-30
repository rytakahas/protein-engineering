from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
import numpy as np

from Bio.PDB import PDBParser, MMCIFParser, PPBuilder


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


def _chain_seq_and_ca(chain) -> Tuple[str, np.ndarray]:
    """
    Returns (sequence_str, coords[L,3]) for a chain.
    Uses PPBuilder to ensure residues & order match between sequence and coords.
    Drops residues missing CA.
    Concatenates multiple peptides within the same chain.
    """
    ppb = PPBuilder()
    seq_parts: List[str] = []
    coords_parts: List[np.ndarray] = []

    for poly in ppb.build_peptides(chain):
        # sequence as string
        seq = str(poly.get_sequence())
        # CA coordinates aligned to the same residues PPBuilder used
        ca_coords: List[List[float]] = []
        for res in poly:
            if "CA" in res:
                ca = res["CA"].get_vector()
                ca_coords.append([float(ca[0]), float(ca[1]), float(ca[2])])
            else:
                # skip residue without CA (also skip from sequence to keep lengths aligned)
                # PPBuilder already omitted most non-standard residues, so rarely hits here
                pass
        if len(ca_coords) == 0 or len(seq) == 0:
            continue
        # Make sure lengths match; if not, trim to min
        L = min(len(seq), len(ca_coords))
        if L == 0:
            continue
        seq_parts.append(seq[:L])
        coords_parts.append(np.asarray(ca_coords[:L], dtype=np.float32))

    if not seq_parts:
        return "", np.zeros((0, 3), dtype=np.float32)

    seq_full = "".join(seq_parts)
    coords_full = np.concatenate(coords_parts, axis=0)
    # Final guard
    L = min(len(seq_full), coords_full.shape[0])
    return seq_full[:L], coords_full[:L]


def _structure_chains(structure) -> Dict[str, Tuple[str, np.ndarray]]:
    """
    Map chain_id -> (sequence, ca_coords[L,3]), skipping empty chains.
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
# Contact map
# --------------------------
def contact_map_from_coords(coords: np.ndarray, thr: float, sym: bool = True):
    """
    coords: [L,3] (float32)
    thr: contact threshold (Å)
    Returns:
      Y: [L,L] float32 {0,1}
      M: [L,L] float32 {0,1} valid-pair mask (all-ones unless NaNs appear)
    """
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        L = int(coords.shape[0]) if coords.ndim >= 1 else 0
        return np.zeros((L, L), dtype=np.float32), np.zeros((L, L), dtype=np.float32)

    # mask invalid rows
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

    # place back into full map
    for i, ii in enumerate(valid):
        for j, jj in enumerate(valid):
            Y[ii, jj] = contact_bool[i, j]
            M[ii, jj] = 1.0

    if sym:
        # ensure symmetry
        Y = np.maximum(Y, Y.T)
        M = np.maximum(M, M.T)

    # zero diagonal (optional but common)
    np.fill_diagonal(Y, 0.0)
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
      - coords:    [coordsA] or [coordsA, coordsB]
      - kind:      "intra" or "inter"
    Stops once `limit` examples have been yielded.
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
            chains = _structure_chains(s)  # dict cid -> (seq, coords[L,3])
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

        if include_inter_chain and len(chains) >= 2:
            # All unordered pairs A<B
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

