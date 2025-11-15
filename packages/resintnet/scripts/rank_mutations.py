#!/usr/bin/env python3
import argparse, os, pandas as pd
from resintnet.rank import top_mutations

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_csv", type=str, default=None, help="If not given, searches outputs/")
    ap.add_argument("--out", type=str, default="./outputs")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    if args.scan_csv is None:
        candidates = [f for f in os.listdir(args.out) if f.startswith("mutational_scan")]
        if not candidates:
            raise SystemExit("No mutational_scan_*.csv found in outputs. Run train_memory.py first.")
        args.scan_csv = os.path.join(args.out, candidates[0])
    df = pd.read_csv(args.scan_csv)
    top = top_mutations(df, args.topk)
    top.to_csv(os.path.join(args.out, "top_mutations.csv"), index=False)
    print(top)

if __name__ == "__main__":
    main()
