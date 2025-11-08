#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, subprocess, shlex, yaml

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run(cmd, dry_run=False, cwd=None):
    print("[CMD]", cmd)
    if dry_run:
        return 0
    proc = subprocess.run(shlex.split(cmd), cwd=cwd)
    return proc.returncode

def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    ap = argparse.ArgumentParser(description="Train ResIntNet GNN (optional)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    root = repo_root()
    py = sys.executable
    P = cfg["paths"]

    # For now call a hypothetical training entrypoint under resintnet (adjust if needed)
    out_dir = os.path.join(P.get('output_dir','outputs'), 'resintnet')
    os.makedirs(out_dir, exist_ok=True)

    cmd = (
        f"{py} packages/resintnet/scripts/rank_mutations.py "
        f"--fasta {P['fasta']} "
        f"--priors {P['template_priors_dir']} "
        f"--emb {P['emb_dir']} "
        f"--msa {P['msa_feat_dir']} "
        f"--train --out {os.path.join(out_dir, 'ranked_train.csv')}"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

if __name__ == "__main__":
    main()
