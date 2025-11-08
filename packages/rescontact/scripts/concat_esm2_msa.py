#!/usr/bin/env python3
"""
Concatenate ESM2 embeddings with MSA-derived features.

Inputs per protein ID:
  - <esm_dir>/<id>.esm2.npy          -> shape (L_esm, C_esm)
  - <msa_dir>/<id>.msa.npz           -> keys: pssm(L,21), coverage(L,), depth(), col2pos(A,), meta(json)

Output per protein:
  - <out_dir>/<id>.esm2_msa.npz      -> keys:
        X      : (L_out, C_esm + C_msa)
        meta   : JSON string with shapes/paths/mode details
        shapes : small array [L_out, C_total] for quick inspection

Default behavior:
  - mode=pad : pad/truncate the MSA features to match L_esm (zero pad tail if needed)
  - include_depth: off by default. If on, adds a broadcast (L,1) channel from scalar depth.

Usage:
  python scripts/concat_esm2_msa.py \
    --esm-dir data/emb/esm2_t12 \
    --msa-dir data/msa_features \
    --out-dir data/emb/esm2_t12_plus_msa \
    --float16 \
    --mode pad \
    --include-depth \
    --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


def load_esm(esm_path: Path) -> np.ndarray:
    arr = np.load(esm_path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"ESM array must be 2D (L,C). Got shape={arr.shape} at {esm_path}")
    return np.array(arr)


def load_msa(msa_path: Path) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Returns (pssm, coverage, depth, meta_json).
    pssm: (L, 21) float
    coverage: (L,) float
    depth: int
    meta_json: str
    """
    z = np.load(msa_path, allow_pickle=False)
    for k in ("pssm", "coverage", "depth", "meta"):
        if k not in z:
            raise KeyError(f"{msa_path} missing key '{k}'")
    pssm = z["pssm"]
    cov = z["coverage"]
    depth = int(z["depth"])
    meta_json = z["meta"].item() if hasattr(z["meta"], "item") else str(z["meta"])
    if pssm.ndim != 2 or pssm.shape[1] != 21:
        raise ValueError(f"{msa_path} pssm must be (L,21). Got {pssm.shape}")
    if cov.ndim != 1 or cov.shape[0] != pssm.shape[0]:
        raise ValueError(f"{msa_path} coverage must be (L,) with same L as pssm. Got {cov.shape}, pssm {pssm.shape}")
    return np.asarray(pssm), np.asarray(cov), depth, meta_json


def make_msa_matrix(pssm: np.ndarray,
                    coverage: np.ndarray,
                    depth: int,
                    include_depth: bool) -> np.ndarray:
    L = pssm.shape[0]
    parts = [pssm, coverage.reshape(L, 1)]
    if include_depth:
        depth_col = np.full((L, 1), float(depth), dtype=pssm.dtype)
        parts.append(depth_col)
    return np.concatenate(parts, axis=1)


def align_lengths(esm: np.ndarray,
                  msa: Optional[np.ndarray],
                  mode: str = "pad") -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Align along sequence length dimension.
    - esm: (L_esm, C_esm)
    - msa: (L_msa, C_msa) or None
    mode:
      - 'pad': keep ESM length; pad/crop MSA to L_esm (zeros for pad)
      - 'crop': crop both to min(L_esm, L_msa)
      - 'strict': require equal lengths or raise
    Returns (esm_aligned, msa_aligned_or_none, info)
    """
    info = {}
    L_esm = esm.shape[0]
    if msa is None:
        info.update({"msa_present": False, "action": "esm_only", "L_esm": L_esm})
        return esm, None, info

    L_msa = msa.shape[0]
    info.update({"msa_present": True, "L_esm": L_esm, "L_msa_raw": L_msa, "mode": mode})

    if mode == "strict":
        if L_esm != L_msa:
            raise ValueError(f"Length mismatch (strict): L_esm={L_esm}, L_msa={L_msa}")
        return esm, msa, info

    if mode == "crop":
        L = min(L_esm, L_msa)
        if L_esm != L:
            esm = esm[:L]
            info["esm_cropped"] = True
        if L_msa != L:
            msa = msa[:L]
            info["msa_cropped"] = True
        info["L_out"] = L
        return esm, msa, info

    # mode == "pad"
    if L_msa > L_esm:
        msa = msa[:L_esm]
        info["msa_cropped"] = True
    elif L_msa < L_esm:
        pad = np.zeros((L_esm - L_msa, msa.shape[1]), dtype=msa.dtype)
        msa = np.concatenate([msa, pad], axis=0)
        info["msa_padded"] = True
    info["L_out"] = L_esm
    return esm, msa, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm-dir", required=True, help="Directory of <id>.esm2.npy files")
    ap.add_argument("--msa-dir", required=True, help="Directory of <id>.msa.npz files (from build_msa_features.py)")
    ap.add_argument("--out-dir", required=True, help="Where to write <id>.esm2_msa.npz")
    ap.add_argument("--esm-suffix", default=".esm2.npy", help="Suffix to strip for IDs (default: .esm2.npy)")
    ap.add_argument("--msa-suffix", default=".msa.npz", help="Suffix for MSA feature files (default: .msa.npz)")
    ap.add_argument("--mode", choices=["pad", "crop", "strict"], default="pad",
                    help="Length alignment strategy (default: pad to ESM length)")
    ap.add_argument("--include-depth", action="store_true",
                    help="Add scalar MSA depth as an extra per-position channel")
    ap.add_argument("--float16", action="store_true", help="Store output X as float16")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    esm_dir = Path(args.esm_dir)
    msa_dir = Path(args.msa_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_out = np.float16 if args.float16 else np.float32
    built, skipped = 0, 0

    esm_files = sorted(esm_dir.glob(f"*{args.esm_suffix}"))
    if not esm_files:
        print(f"[concat] no ESM files matching *{args.esm_suffix} in {esm_dir}", file=sys.stderr)
        sys.exit(1)

    for esm_path in esm_files:
        qid = esm_path.name[:-len(args.esm_suffix)]
        msa_path = msa_dir / f"{qid}{args.msa_suffix}"
        try:
            esm = load_esm(esm_path)   # (L_esm, C_esm)

            if msa_path.exists():
                pssm, cov, depth, meta_json = load_msa(msa_path)
                msa = make_msa_matrix(pssm, cov, depth, include_depth=args.include_depth)  # (L_msa, C_msa)
                msa_present = True
            else:
                msa = None
                msa_present = False
                meta_json = json.dumps({"warning": "msa_missing"})

            esm_aligned, msa_aligned, info = align_lengths(esm, msa, mode=args.mode)

            if msa_aligned is None:
                X = esm_aligned.astype(dtype_out, copy=False)
                C_msa = 0
            else:
                X = np.concatenate([esm_aligned, msa_aligned], axis=1).astype(dtype_out, copy=False)
                C_msa = int(msa_aligned.shape[1])

            meta = {
                "id": qid,
                "esm_path": str(esm_path),
                "msa_path": str(msa_path) if msa_present else None,
                "msa_present": msa_present,
                "mode": args.mode,
                "include_depth": args.include_depth,
                "dtype": "float16" if args.float16 else "float32",
                "L_esm": int(esm.shape[0]),
                "C_esm": int(esm.shape[1]),
                "L_out": int(X.shape[0]),
                "C_out": int(X.shape[1]),
                "C_msa": C_msa,
            }
            meta.update(info)

            out_path = out_dir / f"{qid}.esm2_msa.npz"
            np.savez_compressed(out_path, X=X, meta=json.dumps(meta), shapes=np.array(X.shape, dtype=np.int32))
            built += 1
            if args.verbose:
                print(f"[concat] wrote {out_path.name}  X={tuple(X.shape)}  mode={args.mode}  msa={msa_present}")
        except Exception as e:
            skipped += 1
            print(f"[concat] SKIP {qid}: {e}", file=sys.stderr)

    print(f"[concat] done. built={built} skipped={skipped} out={out_dir}")


if __name__ == "__main__":
    main()

