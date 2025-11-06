#!/usr/bin/env python3
"""
classify_oligomers.py

Scan a directory of PDB/mmCIF files and classify oligomeric state
based on the number of protein chains in the first model.

Requirements:
  pip install biopython

Examples:
  python classify_oligomers.py pdbs/                          # precise names
  python classify_oligomers.py pdbs/ --coarse                 # monomer/dimer/multimer
  python classify_oligomers.py pdbs/ --min-len 20 --csv out.csv
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

from Bio.PDB import PDBParser, MMCIFParser, Polypeptide

# standard 3-letter amino-acid set (Bio.PDB.Polypeptide.is_aa(..., standard=True) uses this)
def _is_protein_res(res) -> bool:
    try:
        return Polypeptide.is_aa(res, standard=True)
    except Exception:
        return False

NAME_MAP: Dict[int, str] = {
    1: "monomer",
    2: "dimer",
    3: "trimer",
    4: "tetramer",
    5: "pentamer",
    6: "hexamer",
    7: "heptamer",
    8: "octamer",
    9: "nonamer",     # (spelling: nonamer)
    10: "decamer",
    11: "undecamer",
    12: "dodecamer",
}

def classify(n: int, coarse: bool) -> str:
    if coarse:
        if n == 1: return "monomer"
        if n == 2: return "dimer"
        return "multimer"
    return NAME_MAP.get(n, f"{n}-mer")

def count_protein_chains(path: Path, min_len: int) -> Tuple[List[str], List[int]]:
    """Return (chain_ids, aa_counts) for chains that have >= min_len protein residues."""
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    struct = parser.get_structure("X", str(path))
    model = next(struct.get_models())  # first model
    chain_ids, aa_counts = [], []
    for chain in model.get_chains():
        aa_count = sum(1 for r in chain.get_residues() if _is_protein_res(r))
        if aa_count >= min_len:
            chain_ids.append(chain.id)
            aa_counts.append(aa_count)
    return chain_ids, aa_counts

def scan_dir(root: Path, recursive: bool, min_len: int, coarse: bool) -> List[Dict]:
    exts = {".pdb", ".ent", ".cif", ".mmcif"}
    files = (root.rglob("*") if recursive else root.glob("*"))
    rows = []
    for f in files:
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        try:
            chain_ids, aa_counts = count_protein_chains(f, min_len=min_len)
            n = len(chain_ids)
            rows.append({
                "file": str(f),
                "n_chains": n,
                "chains": ",".join(chain_ids) if chain_ids else "-",
                "residues_per_chain": ",".join(map(str, aa_counts)) if aa_counts else "-",
                "class": classify(n, coarse=False),
                "coarse_class": classify(n, coarse=True),
            })
        except Exception as e:
            rows.append({
                "file": str(f),
                "n_chains": 0,
                "chains": "-",
                "residues_per_chain": "-",
                "class": "parse_error",
                "coarse_class": "parse_error",
                "error": str(e),
            })
    return rows

def _pct(n: int, denom: int) -> float:
    return (100.0 * n / denom) if denom > 0 else 0.0

def print_summaries(rows: List[Dict]):
    total = len(rows)
    ok_rows = [r for r in rows if r.get("class") != "parse_error"]
    parsed = len(ok_rows)
    errors = total - parsed

    # Coarse summary (monomer / dimer / multimer / parse_error)
    coarse_counts = Counter(r["coarse_class"] for r in rows)
    # Exact class summary (monomer, dimer, trimer, ... or "N-mer")
    exact_counts = Counter(r["class"] for r in rows)

    print("\n--- Summary (denominator = ALL files scanned) ---")
    print(f"Total files scanned: {total}")
    print(f"Parsed successfully: {parsed} ({_pct(parsed, total):.2f}%)")
    print(f"Parse errors:        {errors} ({_pct(errors, total):.2f}%)")

    # Coarse table
    print("\nCoarse classes (count, % of total):")
    for key in ["monomer", "dimer", "multimer", "parse_error"]:
        c = coarse_counts.get(key, 0)
        print(f"  {key:12s}: {c:6d}  {_pct(c, total):6.2f}%")

    # Exact table (sorted: monomer,dimer,trimer,tetramer,..., then others, then parse_error last)
    def sort_key(k: str):
        if k == "parse_error":
            return (9_999, k)
        if k in ("monomer", "dimer", "trimer", "tetramer", "pentamer", "hexamer",
                 "heptamer", "octamer", "nonamer", "decamer", "undecamer", "dodecamer"):
            order = ["monomer", "dimer", "trimer", "tetramer", "pentamer", "hexamer",
                     "heptamer", "octamer", "nonamer", "decamer", "undecamer", "dodecamer"]
            return (order.index(k), k)
        # handle like "13-mer", "20-mer", etc.
        if k.endswith("-mer"):
            try:
                n = int(k.split("-")[0])
                return (100 + n, k)
            except Exception:
                pass
        return (5000, k)

    print("\nExact classes (count, % of total):")
    for cls in sorted(exact_counts.keys(), key=sort_key):
        c = exact_counts[cls]
        print(f"  {cls:12s}: {c:6d}  {_pct(c, total):6.2f}%")

def main():
    ap = argparse.ArgumentParser(description="Classify oligomeric state from chain count (asymmetric unit).")
    ap.add_argument("pdb_dir", type=str, help="Directory containing PDB/mmCIF files.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    ap.add_argument("--min-len", type=int, default=1, help="Minimum AA residues per chain to be counted (default: 1).")
    ap.add_argument("--coarse", action="store_true", help="Also print coarse class (monomer/dimer/multimer).")
    ap.add_argument("--csv", type=str, default="", help="Optional output CSV path (per-file table).")
    args = ap.parse_args()

    root = Path(args.pdb_dir)
    rows = scan_dir(root, args.recursive, args.min_len, args.coarse)

    # Print per-file table to stdout (unchanged)
    header = ["file", "n_chains", "chains", "residues_per_chain", "class", "coarse_class"]
    print(",".join(header))
    for r in rows:
        print(",".join(str(r.get(k, "")) for k in header))

    # Optional CSV
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in header) + "\n")
        print(f"\nWrote per-file table: {out}")

    # Rich summaries (counts + percentages wrt ALL files scanned)
    print_summaries(rows)

if __name__ == "__main__":
    main()

