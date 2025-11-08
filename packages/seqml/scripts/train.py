#!/usr/bin/env python3
from __future__ import annotations
import argparse
from seqml.train_t5_lora import TrainConfig, train_from_csv


def parse():
    ap = argparse.ArgumentParser("Fine-tune T5 (LoRA optional) on mutant efficacy")
    ap.add_argument("--csv", required=True, help="CSV with columns: mut_seq,label")
    ap.add_argument("--model-name", default="google-t5/t5-small")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--out", default="outputs/t5")
    ap.add_argument("--fp16", action="store_true")
    return ap.parse_args()


def main():
    a = parse()
    cfg = TrainConfig(
        model_name=a.model_name,
        lr=a.lr,
        epochs=a.epochs,
        batch_size=a.bs,
        max_len=a.max_len,
        use_lora=not a.no_lora,
        output_dir=a.out,
        fp16=a.fp16,
    )
    out = train_from_csv(a.csv, cfg)
    print(f"[seqml] model saved to {out}")

if __name__ == "__main__":
    main()
