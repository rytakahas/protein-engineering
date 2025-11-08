#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import argparse
from resintnet.rank import rank_one

def parse_args():
    ap = argparse.ArgumentParser("Rank distal mutation hotspots (GNN + PRS blend)")
    ap.add_argument("--priors-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--msa-dir", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--edge-mode", choices=["threshold", "topk"], default="threshold")
    ap.add_argument("--contact-thresh", type=float, default=8.0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    priors_dir = Path(args.priors_dir)
    emb_dir = Path(args.emb_dir)
    msa_dir = Path(args.msa_dir) if args.msa_dir else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for npz in sorted(priors_dir.glob("*.npz")):
        res = rank_one(
            priors_npz=npz,
            emb_dir=emb_dir,
            msa_dir=msa_dir,
            alpha=args.alpha,
            edge_mode=args.edge_mode,
            contact_thresh=args.contact_thresh,
            topk=args.topk,
            device=args.device,
        )
        out_path = out_dir / f"{res['query_id']}.ranking.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[resintnet] wrote {out_path.name}  L={res['L']}  top={res['order'][:5]}")

if __name__ == "__main__":
    main()
