
#!/usr/bin/env python3
import argparse, numpy as np
from src.rescontact.features.fuse import fuse_logits

def main():
    ap = argparse.ArgumentParser(description="Fuse model logits with structural prior.")
    ap.add_argument("--logits-npy", required=True, help="(L,L) logits from the model (before sigmoid)")
    ap.add_argument("--prior-npz", required=True, help="NPZ with 'prior' (L,L) probabilities in [0,1]")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--mode", choices=["logit_blend","prob_blend"], default="logit_blend")
    ap.add_argument("--out-npy", required=True)
    args = ap.parse_args()

    logits = np.load(args.logits_npy)
    prior = np.load(args.prior_npz)["prior"]
    if logits.shape != prior.shape:
        raise ValueError(f"shape mismatch: logits {logits.shape} vs prior {prior.shape}")
    fused = fuse_logits(logits, prior, alpha=args.alpha, mode=args.mode)
    np.save(args.out_npy, fused.astype(np.float32))
    print(f"[fuse_priors] wrote fused logits to {args.out_npy}")

if __name__ == "__main__":
    main()
