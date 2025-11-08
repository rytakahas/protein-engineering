#!/usr/bin/env python3
"""Embed sequences with ESM2 and save .npy per sequence.

Usage:
  python embed_esm2.py --fasta data/fasta/_subset.fa --out-dir data/emb/esm2_t12 --model esm2_t12_35M_UR50D

Deps:
  pip install fair-esm torch biopython
"""

import argparse, numpy as np, torch
from pathlib import Path
from Bio import SeqIO

def choose_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model(model_name: str):
    try:
        import esm
    except ModuleNotFoundError:
        raise SystemExit("[ERROR] 'esm' not found. Install with: pip install fair-esm")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    return model, alphabet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="esm2_t12_35M_UR50D",
                    help="esm2_t6_8M_UR50D | esm2_t12_35M_UR50D | etc.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model, alphabet = load_model(args.model)
    device = choose_device(); model.to(device)
    batch_converter = alphabet.get_batch_converter()

    records = list(SeqIO.parse(args.fasta, "fasta"))
    for rec in records:
        name, sequence = rec.id, str(rec.seq)
        _, _, tokens = batch_converter([(name, sequence)])
        tokens = tokens.to(device)
        with torch.no_grad():
            reps = model(tokens, repr_layers=[model.num_layers])["representations"][model.num_layers][0,1:1+len(sequence)]
        arr = reps.to("cpu").float().numpy()
        np.save(out / f"{name}.esm2.npy", arr)
        print(f"[esm2] wrote {name}.esm2.npy  L={len(sequence)}  C={arr.shape[1]}  device={device}")

if __name__ == "__main__":
    main()
