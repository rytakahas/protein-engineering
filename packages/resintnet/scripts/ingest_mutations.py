#!/usr/bin/env python3
import argparse, os, pandas as pd
from resintnet.ingest.adapters.generic_csv import load_dms_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dms_csv", required=True)
    ap.add_argument("--uni", required=True)
    ap.add_argument("--out", type=str, default="./outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dms = load_dms_csv(args.dms_csv, args.uni)
    df = pd.DataFrame([{"pos":v.pos, "wt":v.wt, "mut":v.mut, "score":v.score} for v in dms.variants])
    df.to_csv(os.path.join(args.out, f"dms_{args.uni}.csv"), index=False)
    print("Saved", os.path.join(args.out, f"dms_{args.uni}.csv"))

if __name__ == "__main__":
    main()
