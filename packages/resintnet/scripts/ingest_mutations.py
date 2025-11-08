#!/usr/bin/env python3
import argparse
from typing import List
import pandas as pd
from resintnet.ingest.adapters import load_d3distal, load_generic_with_mapping
from resintnet.ingest.base import mutations_to_residues

def main():
    ap = argparse.ArgumentParser(description="Ingest distal/allosteric mutation datasets into a normalized schema.")
    ap.add_argument("--source", required=True, help="d3distal | generic")
    ap.add_argument("--in", dest="inputs", required=True, nargs="+", help="input files (CSV/TSV/XLSX)")
    ap.add_argument("--mapping-yaml", default=None, help="YAML mapping (for --source=generic)")
    ap.add_argument("--ref-seq", default=None, help="Reference sequence string (for author indexing)")
    ap.add_argument("--out-mutations", required=True, help="Output CSV with per-mutation rows")
    ap.add_argument("--out-residues", required=True, help="Output CSV with per-residue aggregated labels")
    ap.add_argument("--aggregate", default="any", choices=["any","max","mean"], help="Aggregation for residue labels")
    args = ap.parse_args()

    frames: List[pd.DataFrame] = []
    for path in args.inputs:
        if args.source == "d3distal":
            df = load_d3distal(path=path)
        elif args.source == "generic":
            if not args.mapping_yaml:
                raise SystemExit("--mapping-yaml is required for --source=generic")
            df = load_generic_with_mapping(path=path, mapping_yaml=args.mapping_yaml, ref_seq=args.ref_seq)
        else:
            raise SystemExit(f"Unknown source: {args.source}")
        frames.append(df)

    muts = pd.concat(frames, ignore_index=True).drop_duplicates()
    res = mutations_to_residues(muts, agg=args.aggregate)

    muts.to_csv(args.out_mutations, index=False)
    res.to_csv(args.out_residues, index=False)

if __name__ == "__main__":
    main()
