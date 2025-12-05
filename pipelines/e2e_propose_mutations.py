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
    ap = argparse.ArgumentParser(description="End-to-end: rescontact → resintnet → seqml")
    ap.add_argument("--config", required=True, help="YAML config (see configs/pipeline.example.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    root = repo_root()

    P = cfg["paths"]
    RC = cfg.get("rescontact", {})
    MSA = cfg.get("msa", {})
    RIN = cfg.get("resintnet", {})
    SQ  = cfg.get("seqml", {})

    os.makedirs(P.get("output_dir", "outputs"), exist_ok=True)

    py = sys.executable  # current interpreter

    # 1) Retrieve homologs for template priors
    cmd = (
        f"{py} packages/rescontact/scripts/retrieve_homologs.py "
        f"--fasta {P['fasta']} "
        f"--server-url https://a3m.mmseqs.com "
        f"--max-hits {RC.get('max_hits', 16)} "
        f"--want-templates "
        f"--qps {RC.get('qps', 0.15)} --inter-job-sleep 2 --max-retries 8 --timeout 1800 "
        f"--flush-every 1 "
        f"--out {P['template_hits']}"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

    # 2) Build template priors
    cmd = (
        f"{py} packages/rescontact/scripts/build_template_priors.py "
        f"--hits {P['template_hits']} "
        f"--pdb-root data/pdb/train data/pdb/test "
        f"--out-dir {P['template_priors_dir']} "
        f"--structure-source \"{RC.get('structure_source','pdb,afdb')}\" "
        f"--max-hits-per-query {RC.get('max_hits_per_query',8)} "
        f"--max-downloads-per-run 50"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

    # 3) ESM2 embeddings
    cmd = (
        f"{py} packages/rescontact/scripts/embed_esm2.py "
        f"--fasta {P['fasta']} "
        f"--out-dir {P['emb_dir']} "
        f"--model {RC.get('esm_model','esm2_t12_35M_UR50D')}"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

    # 4) Remote MSA → A3M files
    cmd = (
        f"{py} packages/rescontact/scripts/run_msa_batch.py "
        f"--fasta {P['fasta']} "
        f"--msa-out-dir {P['msa_dir']} "
        f"--server-url {MSA.get('server_url','https://a3m.mmseqs.com')} "
        f"--db {RC.get('db','uniref')} "
        f"--qps {MSA.get('qps',0.15)}"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

    # 5) MSA → numeric features (PSSM/MI/etc.)
    cmd = (
        f"{py} packages/rescontact/scripts/build_msa_features.py "
        f"--msa-dir {P['msa_dir']} "
        f"--esm-emb-dir {P['emb_dir']} "
        f"--out-dir {P['msa_feat_dir']} "
        f"--float16"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

    # 6) Rank mutations (ResIntNet: PRS + (optional) GNN blend)
    out_rank = os.path.join(P.get("output_dir", "outputs"), "ranked_residues.csv")
    cmd = (
        f"{py} packages/resintnet/scripts/rank_mutations.py "
        f"--fasta {P['fasta']} "
        f"--priors {P['template_priors_dir']} "
        f"--emb {P['emb_dir']} "
        f"--msa {P['msa_feat_dir']} "
        f"--alpha {RIN.get('alpha',0.5)} "
        f"--topk {RIN.get('topk_residues',40)} "
        f"--out {out_rank}"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

    # 7) Generate mutants & fine-tune seq model (SeqML)
    out_mut = os.path.join(P.get("output_dir","outputs"), "mutants.csv")
    cmd = (
        f"{py} packages/seqml/scripts/prepare_mutants.py "
        f"--fasta {P['fasta']} "
        f"--ranked {out_rank} "
        f"--out {out_mut} "
        f"--max-mut-distance {SQ.get('mutant_generation',{}).get('max_mut_distance',1)} "
        f"--per-residue-variants {SQ.get('mutant_generation',{}).get('per_residue_variants',5)}"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

    out_dir_seq = os.path.join(P.get("output_dir", "outputs"), "seqml")
    os.makedirs(out_dir_seq, exist_ok=True)
    lora = SQ.get("lora", {})
    cmd = (
        f"{py} packages/seqml/scripts/train.py "
        f"--train-csv {out_mut} "
        f"--model {SQ.get('model_name','t5-small')} "
        f"--epochs {SQ.get('epochs',3)} "
        f"--out-dir {out_dir_seq} "
        f"--lora-r {lora.get('r',8)} "
        f"--lora-alpha {lora.get('alpha',16)} "
        f"--lora-dropout {lora.get('dropout',0.05)}"
    )
    if run(cmd, args.dry_run, cwd=root) != 0: sys.exit(1)

    print("[e2e] Done. Outputs in:", P.get("output_dir","outputs"))

if __name__ == "__main__":
    main()
