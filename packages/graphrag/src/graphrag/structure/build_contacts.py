from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .pdb_utils import extract_ca_coords, extract_chain_sequences, euclidean
from .uniprot_map import map_chain_to_uniprot_positions

def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _read_targets_seq(targets_csv: str) -> Dict[str, str]:
    """Map uniprot_id -> sequence (may be empty)."""
    rows = _read_csv(targets_csv)
    m: Dict[str, str] = {}
    for r in rows:
        uid = (r.get("uniprot_id") or "").strip()
        seq = (r.get("sequence") or "").strip()
        if uid and seq and seq.lower() != "nan":
            m[uid] = seq
    return m

def _kdtree_pairs(coords: List[Tuple[float, float, float]], cutoff: float) -> List[Tuple[int, int, float]]:
    """Return (i,j,dist) for coords within cutoff. Uses SciPy if available; else O(N^2)."""
    try:
        import numpy as np
        from scipy.spatial import cKDTree  # type: ignore

        arr = np.array(coords, dtype=float)
        tree = cKDTree(arr)
        pairs = tree.query_pairs(r=float(cutoff), output_type="set")
        out: List[Tuple[int, int, float]] = []
        for i, j in pairs:
            d = float(np.linalg.norm(arr[i] - arr[j]))
            out.append((int(i), int(j), d))
        return out
    except Exception:
        out: List[Tuple[int, int, float]] = []
        n = len(coords)
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidean(coords[i], coords[j])
                if d <= cutoff:
                    out.append((i, j, d))
        return out

def build_contacts_from_structures_csv(
    *,
    structures_csv: str,
    targets_csv: Optional[str],
    out_csv: str,
    ca_cutoff: float = 8.0,
    weight: float = 1.0,
    contact_type: str = "intra_protein",
) -> int:
    """
    Build an intra-protein residue contact network from each PDB in structures.csv.

    Output columns (compatible with ingest_contacts):
      uniprot_id,chain,i,aa_i,j,aa_j,w,dist,model_id,contact_type

    Notes:
      - i/j are UniProt positions when:
          - targets.csv has sequence for the uniprot_id AND
          - Biopython alignment is installed
      - otherwise: i/j fall back to sequential positions (1..N) along the chosen chain.
    """
    structures = _read_csv(structures_csv)
    seq_map: Dict[str, str] = _read_targets_seq(targets_csv) if targets_csv else {}

    out_rows: List[Dict[str, object]] = []

    for s in structures:
        model_id = (s.get("model_id") or s.get("structure_id") or s.get("pdb_id") or "").strip()
        uniprot_id = (s.get("uniprot_id") or "").strip()
        pdb_path = (s.get("pdb_path") or s.get("path") or "").strip()

        if not model_id or not pdb_path:
            continue
        if not Path(pdb_path).exists():
            continue

        chains = extract_chain_sequences(pdb_path)
        if not chains:
            continue

        # Choose the longest AA chain as "the protein chain" (simple heuristic)
        chosen_chain = max(chains.items(), key=lambda kv: len(str(kv[1].get("sequence", ""))))[0]
        chain_seq = str(chains[chosen_chain].get("sequence", ""))

        # Optional UniProt mapping
        uni_seq = seq_map.get(uniprot_id, "")
        uni_map = None
        if uniprot_id and uni_seq and chain_seq:
            uni_map = map_chain_to_uniprot_positions(
                chain=chosen_chain,
                chain_sequence=chain_seq,
                uniprot_id=uniprot_id,
                uniprot_sequence=uni_seq,
            )

        # CA residues for chosen chain
        recs = [r for r in extract_ca_coords(pdb_path) if r.chain == chosen_chain and r.ca is not None]
        if len(recs) < 2:
            continue

        coords = [r.ca for r in recs if r.ca is not None]  # type: ignore
        pairs = _kdtree_pairs(coords, float(ca_cutoff))

        def pos_for(chain_index0: int) -> int:
            if uni_map is not None:
                return int(uni_map.pdb_index_to_uniprot_pos.get(chain_index0, chain_index0 + 1))
            return chain_index0 + 1

        for i0, j0, dist in pairs:
            ri = recs[i0]
            rj = recs[j0]
            out_rows.append(
                {
                    "uniprot_id": uniprot_id,
                    "chain": chosen_chain,
                    "i": pos_for(i0),
                    "aa_i": ri.aa1,
                    "j": pos_for(j0),
                    "aa_j": rj.aa1,
                    "w": float(weight),
                    "dist": float(dist),
                    "model_id": model_id,
                    "contact_type": contact_type,
                }
            )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["uniprot_id", "chain", "i", "aa_i", "j", "aa_j", "w", "dist", "model_id", "contact_type"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    return len(out_rows)

def main() -> None:
    ap = argparse.ArgumentParser(description="Build residue-residue contacts CSV from PDBs listed in structures.csv")
    ap.add_argument("--structures", required=True, help="data/graphrag/structures.csv")
    ap.add_argument("--targets", required=False, help="data/graphrag/targets.csv (for UniProt sequence mapping)")
    ap.add_argument("--out", required=True, help="output contacts CSV")
    ap.add_argument("--distance", type=float, default=8.0, help="CA-CA cutoff (Å)")
    ap.add_argument("--weight", type=float, default=1.0, help="contact weight")
    ap.add_argument("--contact-type", default="intra_protein", help="tag to store on relationship")
    args = ap.parse_args()

    n = build_contacts_from_structures_csv(
        structures_csv=args.structures,
        targets_csv=args.targets,
        out_csv=args.out,
        ca_cutoff=float(args.distance),
        weight=float(args.weight),
        contact_type=str(args.contact_type),
    )
    print(f"✅ wrote {args.out} with {n} contacts")

if __name__ == "__main__":
    main()
