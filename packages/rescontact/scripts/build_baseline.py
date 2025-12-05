#!/usr/bin/env python3
import argparse, json, random, time
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, default_collate

# --- repo paths ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rescontact.data.dataset import PDBContactDataset

# Reuse tiny utilities from your training stack
# (SimpleContactNet, align_embed_dim are required; collate_skip_none is optional with fallback)
from scripts.train import SimpleContactNet, align_embed_dim
try:
    from scripts.train import collate_skip_none  # preferred (drops None samples)
except Exception:
    # --- Fallback: drop None samples & rows missing core tensors ---
    def _sanitize_sample(sample):
        if sample is None:
            return None
        out = dict(sample)
        if out.get("msa_path") is None:
            out["msa_path"] = ""
        for k in ("emb", "contacts", "mask"):
            if out.get(k, None) is None:
                return None
        return out
    def collate_skip_none(batch):
        batch = [_sanitize_sample(b) for b in batch if b is not None]
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return default_collate(batch)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def upper_tri_ut(L: int, device=None):
    """Upper-triangular indices (i<j) for LxL matrices (torch if device given, else numpy)."""
    if device is None:
        return np.triu_indices(L, k=1)
    return torch.triu_indices(L, L, offset=1, device=device)

def to_np(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)

def compute_hist(vals: np.ndarray, bins):
    h, _ = np.histogram(vals, bins=bins)
    s = h.sum()
    return (h / s).tolist() if s > 0 else [0.0] * (len(bins) - 1)

def main():
    ap = argparse.ArgumentParser(description="Build PSI baseline json for rescontact.")
    ap.add_argument("--config", required=True, help="YAML config used for training.")
    ap.add_argument("--split", choices=["train","val"], default="train",
                    help="Which split to sample for baseline.")
    ap.add_argument("--max-examples", type=int, default=900,
                    help="Max sequences to include (after split).")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    # Bin edges (editable)
    ap.add_argument("--score-bins",
        default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0")
    ap.add_argument("--sep-bins", default="0,6,12,24,48,96,192,384,10000")
    ap.add_argument("--len-bins", default="0,150,300,450,600,900,1200,2000")

    # Model usage
    ap.add_argument("--use-checkpoint", action="store_true",
                    help="If set, load ckpt_dir/model_best.pt and baseline PREDICTED probabilities.")
    ap.add_argument("--out", default="monitor/baseline.json")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))

    # Device
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    # Parse bins
    score_bins = [float(x) for x in args.score_bins.split(",")]
    sep_bins   = [int(x)   for x in args.sep_bins.split(",")]
    len_bins   = [int(x)   for x in args.len_bins.split(",")]

    # Dataset
    ds_full = PDBContactDataset(
        root_dir=cfg["paths"]["train_dir"],
        cache_dir=cfg["paths"]["cache_dir"],
        contact_threshold=cfg["labels"]["contact_threshold_angstrom"],
        include_inter_chain=cfg["labels"]["include_inter_chain"],
        esm_model_name=cfg["model"]["esm_model"],
        use_msa=bool(cfg["features"]["use_msa"]),
        msa_cfg=cfg["features"]["msa"],
    )

    # Reproducible shuffle, then split like training
    idx = list(range(len(ds_full)))
    random.Random(cfg["project"].get("seed", args.seed)).shuffle(idx)
    frac = float(cfg["training"]["train_val_split"])
    n_tr = max(1, int(len(ds_full) * frac))
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
    use_idx = tr_idx if args.split == "train" else va_idx

    if args.max_examples is not None:
        use_idx = use_idx[:args.max_examples]

    ds = Subset(ds_full, use_idx)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_skip_none,
    )

    # Optional model for predicted prob baseline
    want_dim = int(cfg["model"]["embed_dim"])
    model = None
    if args.use_checkpoint:
        model = SimpleContactNet(
            embed_dim=want_dim,
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            dist_bias_max=int(cfg["model"]["distance_bias_max"]),
            dropout_p=float(cfg["model"].get("dropout_p", 0.1)),
        ).to(device).eval()
        ckpt = Path(cfg["paths"]["ckpt_dir"]) / "model_best.pt"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"], strict=False)
            print(f"[baseline] loaded checkpoint: {ckpt}")
        else:
            print(f"[baseline] WARN: {ckpt} not found; falling back to label prevalence for scores.")
            model = None

    # Accumulators
    all_scores = []      # predicted probs (preferred) or label prevalence if no model
    all_seps = []        # |i-j| for positive contacts
    all_lengths = []     # L per example
    all_emb_norms = []   # per-residue ||emb||
    all_msa_cov = []     # per-seq coverage (0..1)

    with torch.no_grad():
        for batch in loader:
            if batch is None:   # dropped batch (all samples invalid)
                continue

            emb = batch["emb"].to(device).float()      # [B,L,D]
            y   = batch["contacts"].to(device).float() # [B,L,L]
            m   = batch["mask"].to(device).float()     # [B,L,L]
            emb = align_embed_dim(emb, want_dim)

            B, L, D = emb.shape
            i_ut, j_ut = upper_tri_ut(L, device=emb.device)  # <<< use sequence length, not D!

            # (seq length)
            all_lengths.extend([L] * B)

            # (embedding norms)
            emb_norms = torch.linalg.vector_norm(emb, dim=-1)  # [B,L]
            all_emb_norms.append(emb_norms.flatten().cpu().numpy())

            # (msa coverage)
            if D >= 341:
                last21 = emb[..., -21:]
                nz = (last21 != 0).sum(dim=(1, 2)).float()
                cov = (nz / float(21 * L)).cpu().numpy()  # [B]
            else:
                cov = np.zeros(B, dtype=np.float32)
            all_msa_cov.append(cov)

            # Scores & distances
            if model is not None:
                logits = model(emb)                      # [B,L,L]
                probs  = torch.sigmoid(logits)          # [B,L,L]
                for b in range(B):
                    mb = m[b][i_ut, j_ut] > 0.5
                    if mb.sum().item() == 0:
                        continue
                    pv = probs[b][i_ut, j_ut][mb].cpu().numpy()
                    all_scores.append(pv)

                    yy = (y[b][i_ut, j_ut] > 0.5) & mb
                    pd = (j_ut - i_ut)[yy].detach().cpu().numpy().astype(np.int32)
                    if pd.size:
                        all_seps.append(pd)
            else:
                # Fallback: use label prevalence as the "score" baseline
                for b in range(B):
                    mb = m[b][i_ut, j_ut] > 0.5
                    if mb.sum().item() == 0:
                        continue
                    yv = y[b][i_ut, j_ut][mb].detach().cpu().numpy().astype(np.float32)
                    all_scores.append(yv)

                    yy = (y[b][i_ut, j_ut] > 0.5) & mb
                    pd = (j_ut - i_ut)[yy].detach().cpu().numpy().astype(np.int32)
                    if pd.size:
                        all_seps.append(pd)

    # Concatenate buffers
    def cat(lst):
        return np.concatenate(lst) if len(lst) else np.array([], dtype=np.float32)

    scores  = cat(all_scores)
    seps    = cat(all_seps)
    lengths = np.array(all_lengths, dtype=np.int32)
    embnorm = cat(all_emb_norms)
    msacov  = cat(all_msa_cov)

    if scores.size == 0:
        raise SystemExit("No valid pairs encountered; check masks / dataset.")

    # Histograms
    baseline = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": f"split={args.split}, max_examples={args.max_examples}, "
                 f"use_checkpoint={bool(model is not None)}, cfg={Path(args.config).name}",
        "bins": {
            "score": score_bins,
            "sep":   sep_bins,
            "length": len_bins,
            "emb_norms": 10,        # DriftMonitor will bin into 10
            "msa_coverage": 10,
        },
        "expected": {
            "score":   compute_hist(scores, score_bins),
            "sep":     compute_hist(seps,   sep_bins),
            "length":  compute_hist(lengths, len_bins),
            # For these two, we store raw samples and let DriftMonitor bin later
            "emb_norms_samples": embnorm.tolist(),
            "msa_coverage_samples": msacov.tolist(),
        },
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"[baseline] wrote {out_path} "
          f"(score N={scores.size}, sep N={seps.size}, len N={lengths.size}, "
          f"emb_norms N={embnorm.size}, msa_cov N={msacov.size})")

if __name__ == "__main__":
    main()

