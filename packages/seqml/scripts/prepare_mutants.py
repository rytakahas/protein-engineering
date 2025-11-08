#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from seqml.mutgen import save_mutant_csv


def parse():
    ap = argparse.ArgumentParser("Generate point mutants CSV from ResIntNet ranking")
    ap.add_argument("--query-id", required=True)
    ap.add_argument("--wt-fasta", required=True, help="FASTA file with a single sequence")
    ap.add_argument("--ranking-json", required=True, help="resintnet JSON (…/rankings/{qid}.ranking.json)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--per-pos-k", type=int, default=5)
    return ap.parse_args()


def read_fasta_one(path: str) -> str:
    txt = Path(path).read_text().splitlines()
    seq = "".join([ln.strip() for ln in txt if not ln.startswith(">")])
    return seq


def main():
    args = parse()
    wt = read_fasta_one(args.wt_fasta)
    save_mutant_csv(args.query_id, wt, Path(args.ranking_json), Path(args.out_csv), per_pos_k=args.per_pos_k)
    print(f"[seqml] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
