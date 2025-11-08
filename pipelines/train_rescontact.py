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
    ap = argparse.ArgumentParser(description="Train ResContact models")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    root = repo_root()
    py = sys.executable

    P = cfg["paths"]
    cmd = (
        f"{py} packages/rescontact/scripts/train.py "
        f"--fasta {P['fasta']} "
        f"--emb-dir {P['emb_dir']} "
        f"--msa-dir {P['msa_dir']} "
        f"--priors {P['template_priors_dir']} "
        f"--out-dir {os.path.join(P.get('output_dir','outputs'), 'rescontact')}"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

if __name__ == "__main__":
    main()
