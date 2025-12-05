#!/usr/bin/env python3

import argparse
import pathlib
import sys
from typing import List, Tuple

# Make sure we can import mmseqs_msa_client.py that sits in the same folder
THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))

from mmseqs_msa_client import run_batch  # expects run_batch(seqs, out_dir, server, db, qps)


def read_fasta(path: str) -> List[Tuple[str, str]]:
    seqs: List[Tuple[str, str]] = []
    with open(path) as f:
        hid = None
        buf = []
        for line in f:
            if line.startswith(">"):
                if hid is not None:
                    seqs.append((hid, "".join(buf).replace(" ", "").replace("\n", "")))
                hid = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if hid is not None:
            seqs.append((hid, "".join(buf)))
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", required=True)
    p.add_argument("--msa-out-dir", required=True)
    p.add_argument("--server-url", default="https://a3m.mmseqs.com")
    p.add_argument("--db", default="uniref")
    p.add_argument("--qps", type=float, default=0.15, help="Queries per second (throttle)")
    args = p.parse_args()

    out_dir = pathlib.Path(args.msa_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = read_fasta(args.fasta)
    if not seqs:
        print(f"[run_msa_batch] No sequences found in {args.fasta}")
        return

    print(f"[run_msa_batch] sequences: {len(seqs)}  server={args.server_url}  db={args.db}  qps={args.qps}")
    run_batch(
        seqs=seqs,
        out_dir=str(out_dir),
        server=args.server_url,
        db=args.db,
        qps=args.qps,
    )
    print("[run_msa_batch] done.")


if __name__ == "__main__":
    main()

