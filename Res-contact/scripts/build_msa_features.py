#!/usr/bin/env python3
"""
Build simple MSA-derived features from A3M files.

Outputs per protein (id.msa.npz):
  - pssm:      (L, 21) float{16,32}, 20 AAs + 1 "other"
  - coverage:  (L,)    float{16,32}, fraction of non-gaps per position
  - depth:     ()      int, number of sequences used (including query)
  - col2pos:   (A,)    int, alignment column -> query position (0..L-1), -1 for gap
  - meta:      str, JSON with info

We follow A3M convention: lowercase letters are insertions and are REMOVED from all sequences
before building the alignment and features.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical
AA_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}
OTHER_INDEX = 20  # 21st channel for non-canonical (B, Z, X, U, O, etc.)


def _read_a3m(path: Path) -> List[Tuple[str, str]]:
    """Read A3M as list of (header, seq) without newlines; empty lines ignored."""
    records: List[Tuple[str, str]] = []
    hid = None
    buf: List[str] = []
    with path.open() as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if hid is not None:
                    records.append((hid, "".join(buf)))
                hid = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
        if hid is not None:
            records.append((hid, "".join(buf)))
    return records


def _strip_insertions(seq: str) -> str:
    """Remove lowercase (insertions) per A3M convention."""
    # keep gaps '-' and uppercase letters; drop lowercase
    return "".join(ch for ch in seq if (ch == "-" or (ch.isalpha() and ch.isupper())))


def _clean_alignment(records: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Apply _strip_insertions to all sequences and drop rows that mismatch length."""
    if not records:
        return []
    # Clean all
    cleaned = [(h, _strip_insertions(s)) for (h, s) in records]
    # Ensure all are same length as first
    L0 = len(cleaned[0][1])
    aligned = [(h, s) for (h, s) in cleaned if len(s) == L0]
    return aligned


def _col2pos_from_query(aligned_query: str) -> np.ndarray:
    """
    Build alignment-column -> query-position map.
    - Returns int array of length = alignment width.
    - For a query gap column, value is -1.
    - Positions are 0..(L-1) where L = number of query residues (non-gaps).
    """
    col2pos = np.full(len(aligned_query), -1, dtype=np.int32)
    pos = 0
    for col, ch in enumerate(aligned_query):
        if ch == "-":
            continue
        # ch should be uppercase A-Z (possibly non-canonical); still advances position
        col2pos[col] = pos
        pos += 1
    return col2pos


def _expect_L_from_esm(emb_dir: Path, qid: str) -> int:
    """
    Try to get expected L from an ESM2 embedding .npy written as <id>.esm2.npy
    Return -1 if not found.
    """
    npy = emb_dir / f"{qid}.esm2.npy"
    if npy.exists():
        try:
            arr = np.load(npy, mmap_mode="r")
            return int(arr.shape[0])
        except Exception:
            pass
    return -1


def a3m_to_features(a3m_path: Path, expect_L: int = -1, dtype=np.float32):
    """
    Convert one A3M into per-position features:
      - pssm: (L, 21)
      - coverage: (L,)
      - depth: int
      - col2pos: (A,)  alignment columns -> positions
      - meta: dict
    """
    recs = _read_a3m(a3m_path)
    if not recs:
        raise ValueError(f"{a3m_path}: empty A3M")

    aligned = _clean_alignment(recs)
    if not aligned:
        raise ValueError(f"{a3m_path}: no valid aligned rows after cleaning")

    # Query = first entry
    qh, q_aln = aligned[0]
    A = len(q_aln)  # alignment width after stripping insertions

    col2pos = _col2pos_from_query(q_aln)
    L_from_query = int((col2pos >= 0).sum())

    # Decide final L
    L = L_from_query
    if expect_L > 0 and expect_L != L_from_query:
        # We will keep only columns whose mapped position is within [0, expect_L)
        # and then set L=expect_L (padding rows if necessary).
        L = expect_L

    # Gather usable columns: those that map to a valid position in range
    valid_cols = np.where(col2pos >= 0)[0]
    if expect_L > 0:
        # filter out columns that would map beyond expect_L
        valid_cols = [c for c in valid_cols if col2pos[c] < expect_L]
    if len(valid_cols) == 0:
        raise ValueError(f"{a3m_path}: no valid columns after mapping (expect_L={expect_L}, Lq={L_from_query})")

    # Initialize accumulators
    pssm = np.zeros((L, 21), dtype=dtype)
    cov = np.zeros((L,), dtype=dtype)

    # Count rows (including query)
    depth = 0

    for (hdr, seq_aln) in aligned:
        if len(seq_aln) != A:
            # Shouldn't happen because _clean_alignment enforced length match, but guard anyway.
            continue
        depth += 1
        for c in valid_cols:
            pos = int(col2pos[c])  # 0..Lq-1
            if pos < 0 or pos >= L:
                continue
            aa = seq_aln[c]
            if aa == "-":
                # gap does not contribute to amino-count; affects coverage later
                continue
            # Uppercase ensured; map to index
            idx = AA_INDEX.get(aa, OTHER_INDEX)
            pssm[pos, idx] += 1.0
            cov[pos] += 1.0  # count non-gap in this row at this position

    # Normalize PSSM to frequencies (avoid div-by-zero)
    # Note: coverage is number of non-gaps per position across 'depth' rows
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = cov.copy()
        denom[denom == 0] = 1.0
        pssm = pssm / denom[:, None]

    # Convert coverage to fraction (0..1)
    if depth > 0:
        cov = cov / float(depth)

    meta = {
        "a3m": str(a3m_path),
        "query_header": qh,
        "alignment_cols": A,
        "L_from_query": L_from_query,
        "L_final": L,
        "depth": depth,
        "valid_cols": int(len(valid_cols)),
        "expect_L": expect_L,
        "channels": {"AA_ORDER": AA_ORDER, "OTHER_INDEX": OTHER_INDEX},
    }

    return {
        "pssm": pssm.astype(dtype, copy=False),
        "coverage": cov.astype(dtype, copy=False),
        "depth": np.array(depth, dtype=np.int32),
        "col2pos": col2pos.astype(np.int32, copy=False),
        "meta": json.dumps(meta),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa-dir", required=True, help="Directory with *.a3m files")
    ap.add_argument("--esm-emb-dir", required=False, default=None,
                    help="Directory with <id>.esm2.npy to infer L; optional")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--float16", action="store_true", help="Store features in float16")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    msa_dir = Path(args.msa_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.esm_emb_dir) if args.esm_emb_dir else None

    dtype = np.float16 if args.float16 else np.float32

    a3ms = sorted(msa_dir.glob("*.a3m"))
    if not a3ms:
        print(f"[msa_features] no .a3m in {msa_dir}", file=sys.stderr)
        sys.exit(1)

    built = 0
    for a3m in a3ms:
        qid = a3m.stem  # matches your run_msa_batch output (e.g., 106M_A.a3m)
        try:
            expect_L = -1
            if emb_dir:
                expect_L = _expect_L_from_esm(emb_dir, qid)
            feats = a3m_to_features(a3m, expect_L=expect_L, dtype=dtype)
            npz_path = out_dir / f"{qid}.msa.npz"
            np.savez_compressed(npz_path, **feats)
            built += 1
            if args.verbose:
                meta = json.loads(feats["meta"])
                print(f"[msa_features] wrote {npz_path.name}  L={meta['L_final']}  depth={meta['depth']}")
        except Exception as e:
            print(f"[msa_features] SKIP {qid}: {e}", file=sys.stderr)

    print(f"[msa_features] done. built={built} out={out_dir}")


if __name__ == "__main__":
    main()

