#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build template distance priors from MMseqs PDB template hits.

Inputs
------
--hits: JSON from scripts/retrieve_homologs.py (may NOT contain "sequence")
--pdb-root: one or more roots containing query PDB files (e.g., data/pdb/train data/pdb/test)
--out-dir: output directory for priors (NPZ per query)
--structure-source: comma list (pdb,afdb) [afdb is ignored unless you add a fetcher]
--max-hits-per-query: int, cap number of template hits used per query
--max-downloads-per-run: int, cap number of PDB downloads (templates not present locally)

What it does
------------
1) For each query in hits JSON:
   - Obtain query sequence:
       a) Use q["sequence"] if present
       b) Else derive from local PDB SEQRES (preferred)
       c) Else derive from local PDB ATOMs
   - L = len(query sequence)
2) For each PDB template hit (subject looks like '1abc_A' or '123L_A'):
   - Resolve/get template PDB path (local or download from RCSB if allowed)
   - Build template ATOM-based sequence (ensures 1:1 with CA coordinates)
   - Global align (Bio.Align.PairwiseAligner) query_seq <-> template_atom_seq
   - Use alignment to map query indices -> template residue indices
   - Compute Cα–Cα distances on the mapped template positions
   - Convert distances to one-hot over distance bins; accumulate into priors[L,L,B]
3) Average priors across templates; save NPZ with: priors, bins, mask, meta

Notes
-----
- We treat only positions where both mapped residues have CA.
- Distance bins: [0,4,6,8,10,12,14,16,20,24,30] (10 bins).
- Symmetrize priors (i,j) and (j,i).
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import math
import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser
from Bio import Align

# -----------------------
# Configuration defaults
# -----------------------
DIST_BIN_EDGES = [0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 30.0]  # 10 bins
CA_ATOM = "CA"

# Conservative protein scoring (global)
ALIGNER = Align.PairwiseAligner()
ALIGNER.mode = "global"
ALIGNER.match_score = 2.0
ALIGNER.mismatch_score = -1.0
ALIGNER.open_gap_score = -10.0
ALIGNER.extend_gap_score = -0.5

# 3-letter to 1-letter mapping (robust, no Biopython 3to1 dependency)
AA3_TO_1 = {
    # Standard
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Selenomethionine and common alts -> M
    "MSE": "M",
    # Ambiguous / unknown -> X
    "ASX": "B", "GLX": "Z", "SEC": "U", "PYL": "O",
}

# Regex to parse IDs
PDB_SUBJECT_RE = re.compile(r"^([0-9][A-Za-z0-9]{3}|[0-9]{3}[A-Za-z])[_-]([A-Za-z0-9])$")  # e.g., 1abc_A or 106M_A

# -----------------------
# Utilities
# -----------------------

def _toupper(s: str) -> str:
    return s.upper() if s is not None else s

def bin_index(d: float, edges: List[float]) -> int:
    # right-open bins [e_i, e_{i+1})
    for i in range(len(edges) - 1):
        if edges[i] <= d < edges[i + 1]:
            return i
    return len(edges) - 2  # clamp to last bin if >= last edge

def find_pdb_path(pdb_roots: List[Path], pdb_code: str) -> Optional[Path]:
    """Find {pdb_code}.pdb in any root, case-insensitive."""
    candidates = {
        f"{pdb_code}.pdb",
        f"{pdb_code.upper()}.pdb",
        f"{pdb_code.lower()}.pdb",
    }
    for root in pdb_roots:
        for c in candidates:
            p = root / c
            if p.exists():
                return p
    return None

def parse_seqres_sequence(pdb_path: Path, chain_id: str) -> Optional[str]:
    """Parse SEQRES lines for the chain; returns string or None if not found."""
    seq = []
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith("SEQRES"):
                    continue
                ch = line[11].strip()
                if ch != chain_id:
                    continue
                # Residue names are columns 18-70 split by spaces (PDB format)
                # Safer: split everything beyond col 19
                parts = line[19:].split()
                for res3 in parts:
                    res1 = AA3_TO_1.get(res3.upper(), "X")
                    seq.append(res1)
        return "".join(seq) if seq else None
    except Exception:
        return None

def parse_atom_sequence_and_calpha(pdb_path: Path, chain_id: str) -> Tuple[str, List[Optional[np.ndarray]]]:
    """Extract ATOM-based sequence and CA coords for chain."""
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure("x", str(pdb_path))
    chain = None
    for model in structure:
        for ch in model:
            if _toupper(ch.id) == _toupper(chain_id):
                chain = ch
                break
        if chain:
            break
    if chain is None:
        return "", []

    seq = []
    ca_coords: List[Optional[np.ndarray]] = []
    # Iterate residues in order; keep only standard residues
    for res in chain.get_residues():
        resname = res.get_resname().upper() if hasattr(res, "get_resname") else None
        if resname is None:
            continue
        aa = AA3_TO_1.get(resname, "X")
        # Some HetAtoms (e.g., HOH) -> skip unless amino acid
        if aa == "X" and res.id[0].strip() != " ":
            continue
        # record
        seq.append(aa)
        if CA_ATOM in res:
            ca_coords.append(np.array(list(res[CA_ATOM].get_vector())))
        else:
            ca_coords.append(None)
    return "".join(seq), ca_coords

def global_align_map(q: str, t: str) -> List[Tuple[int, int]]:
    """
    Return mapping list of (qi, tj) 0-indexed positions where both are residues (not gaps).
    """
    if not q or not t:
        return []

    # Align; PairwiseAligner returns Alignment objects with coordinates
    # Use the first/best alignment (global)
    aln = max(ALIGNER.align(q, t), key=lambda a: a.score)

    q_aln = aln.aligned[0]  # list of (start, end) blocks in query
    t_aln = aln.aligned[1]  # list of (start, end) blocks in template
    mapping: List[Tuple[int, int]] = []
    for (qs, qe), (ts, te) in zip(q_aln, t_aln):
        # block of matches/mismatches (no gaps inside)
        for i in range(qs, qe):
            j = ts + (i - qs)
            mapping.append((i, j))
    return mapping

def subject_to_pdb_chain(subject: str) -> Optional[Tuple[str, str]]:
    """
    Convert subject string to (pdb_code, chain_id).
    Accepts forms like '1p2r_A', '106M_A', '1ABC-A'.
    """
    s = subject.replace("-", "_")
    m = PDB_SUBJECT_RE.match(s)
    if not m:
        return None
    pdb_code, chain = m.group(1), m.group(2)
    return pdb_code.upper(), chain.upper()

def download_pdb_to(path: Path, pdb_code: str) -> bool:
    """
    Download PDB file (PDB format) from RCSB and save as path.
    Tries both .pdb and .pdb.gz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    url_txt = f"https://files.rcsb.org/view/{pdb_code}.pdb"
    url_gz = f"https://files.rcsb.org/view/{pdb_code}.pdb.gz"
    try:
        with urllib.request.urlopen(url_txt, timeout=60) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
        return True
    except Exception:
        pass
    try:
        gz_path = path.with_suffix(path.suffix + ".gz")
        with urllib.request.urlopen(url_gz, timeout=60) as r, open(gz_path, "wb") as f:
            shutil.copyfileobj(r, f)
        # gunzip
        with gzip.open(gz_path, "rb") as gz, open(path, "wb") as out:
            shutil.copyfileobj(gz, out)
        gz_path.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def make_priors(L: int, bins: List[float]) -> np.ndarray:
    B = len(bins) - 1
    return np.zeros((L, L, B), dtype=np.float32)

def normalize_priors(priors: np.ndarray) -> np.ndarray:
    # priors: (L, L, B)
    sums = priors.sum(axis=-1, keepdims=True)  # (L, L, 1)
    # Safe in-place division only where sums>0 (broadcasted), leaves zeros otherwise
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(priors, sums, out=priors, where=(sums > 0))
    return priors

def build_for_query(
    qid: str,
    q_seq: str,
    hits: List[Dict],
    pdb_roots: List[Path],
    out_dir: Path,
    max_hits_per_query: int,
    max_downloads: List[int],
    allow_sources: set,
) -> Optional[Path]:
    """
    Build priors for one query; returns NPZ path or None.
    max_downloads is a single-item list used as mutable counter.
    """
    qL = len(q_seq)
    if qL == 0:
        print(f"[priors] {qid}: empty query sequence -> skip")
        return None

    priors = make_priors(qL, DIST_BIN_EDGES)
    used_templates = 0

    # Where to cache downloaded PDBs
    cache_dir = out_dir / "_cache_pdb"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Filter hits to PDB-only templates with alignment fields present or at least plausible
    pdb_hits: List[Tuple[str, Dict]] = []
    for h in hits:
        subj = h.get("subject")
        if not subj:
            continue
        parsed = subject_to_pdb_chain(subj)
        if not parsed:
            continue
        pdb_hits.append((subj, h))

    if not pdb_hits:
        print(f"[priors] {qid}: no PDB template hits -> skip")
        return None

    # Cap number of templates
    pdb_hits = pdb_hits[:max_hits_per_query]

    for subj, h in pdb_hits:
        parsed = subject_to_pdb_chain(subj)
        if not parsed:
            continue
        pdb_code, chain_id = parsed

        # Find or download template PDB
        t_path = find_pdb_path(pdb_roots, pdb_code)
        if t_path is None:
            if "pdb" not in allow_sources:
                # can't fetch
                print(f"[priors] {qid}: template {subj} missing locally and 'pdb' not allowed -> skip")
                continue
            if max_downloads[0] <= 0:
                print(f"[priors] {qid}: download budget exhausted; skipping remaining templates")
                break
            t_path = cache_dir / f"{pdb_code}.pdb"
            if not t_path.exists():
                ok = download_pdb_to(t_path, pdb_code)
                if not ok:
                    print(f"[priors] {qid}: failed to download PDB {pdb_code} -> skip")
                    continue
                max_downloads[0] -= 1

        # Build template ATOM-based sequence and CA coords (same length)
        t_seq, t_ca = parse_atom_sequence_and_calpha(t_path, chain_id)
        if not t_seq:
            print(f"[priors] {qid}: template {subj} ATOM-seq empty -> skip")
            continue
        # Map q indices -> t indices via global alignment
        mapping = global_align_map(q_seq, t_seq)
        if not mapping:
            print(f"[priors] {qid}: no alignment mapping with {subj} -> skip")
            continue

        # For each pair (i,j) in query mapped to (ti,tj), add one-hot distance bin
        B = len(DIST_BIN_EDGES) - 1
        for a in range(len(mapping)):
            qi, ti = mapping[a]
            if ti < 0 or ti >= len(t_ca):
                continue
            ci = t_ca[ti]
            if ci is None:
                continue
            for b in range(a + 1, len(mapping)):
                qj, tj = mapping[b]
                if tj < 0 or tj >= len(t_ca):
                    continue
                cj = t_ca[tj]
                if cj is None:
                    continue
                d = float(np.linalg.norm(ci - cj))
                bidx = bin_index(d, DIST_BIN_EDGES)
                priors[qi, qj, bidx] += 1.0
                priors[qj, qi, bidx] += 1.0

        used_templates += 1

    if used_templates == 0:
        print(f"[priors] {qid}: no usable templates -> skip")
        return None

    priors = normalize_priors(priors)
    mask = (priors.sum(axis=-1) > 0).astype(np.uint8)

    out_path = out_dir / f"{qid}.npz"
    np.savez_compressed(
        out_path,
        priors=priors,
        bins=np.array(DIST_BIN_EDGES, dtype=np.float32),
        mask=mask,
        meta=np.string_(json.dumps({"query_id": qid, "L": len(q_seq), "templates_used": used_templates}))
    )
    print(f"[priors] wrote {out_path.name}  L={len(q_seq)}  templates={used_templates}")
    return out_path

def get_query_sequence(q: Dict, pdb_roots: List[Path]) -> Tuple[str, Optional[Tuple[str, str]]]:
    """
    Get query sequence. Returns (sequence, (pdb_code, chain)) if derived from PDB id.
    """
    if "sequence" in q and q["sequence"]:
        return q["sequence"], None

    qid = q.get("query_id", "")
    parsed = subject_to_pdb_chain(qid)
    if parsed:
        pdb_code, chain = parsed
        # Prefer SEQRES
        p = find_pdb_path(pdb_roots, pdb_code)
        if p:
            seq = parse_seqres_sequence(p, chain)
            if seq and len(seq) > 0:
                return seq, (pdb_code, chain)
            # Fallback ATOM sequence
            atom_seq, _ = parse_atom_sequence_and_calpha(p, chain)
            if atom_seq and len(atom_seq) > 0:
                return atom_seq, (pdb_code, chain)
    # If we cannot infer from PDB, last resort: build X * length
    L = q.get("length", 0) or 0
    if L:
        return "X" * int(L), None
    return "", None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True, help="JSON from retrieve_homologs.py")
    ap.add_argument("--pdb-root", nargs="+", required=True, help="PDB roots for query PDBs (and possibly templates)")
    ap.add_argument("--out-dir", required=True, help="Directory to write priors (*.npz)")
    ap.add_argument("--structure-source", default="pdb,afdb", help="Comma list: pdb,afdb (afdb currently ignored)")
    ap.add_argument("--max-hits-per-query", type=int, default=8)
    ap.add_argument("--max-downloads-per-run", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_roots = [Path(r) for r in args.pdb_root]
    sources = set([s.strip().lower() for s in args.structure_source.split(",") if s.strip()])

    with open(args.hits, "r") as fh:
        hits = json.load(fh)

    queries = hits.get("queries", [])
    if not queries:
        print("[priors] no queries found in hits json")
        sys.exit(0)

    # mutable counter holder for downloads left
    downloads_left = [int(args.max_downloads_per_run)]

    built = 0
    for q in queries:
        qid = q.get("query_id", "unknown")
        q_seq, from_pdb = get_query_sequence(q, pdb_roots)
        if not q_seq:
            print(f"[priors] {qid}: cannot determine query sequence -> skip")
            continue

        npz_path = build_for_query(
            qid=qid,
            q_seq=q_seq,
            hits=q.get("hits", [])[: args.max_hits_per_query],
            pdb_roots=pdb_roots,
            out_dir=out_dir,
            max_hits_per_query=args.max_hits_per_query,
            max_downloads=downloads_left,
            allow_sources=sources,
        )
        if npz_path is not None:
            built += 1

    print(f"[priors] done. built={built} npz, downloads_remaining={downloads_left[0]}")

if __name__ == "__main__":
    main()

