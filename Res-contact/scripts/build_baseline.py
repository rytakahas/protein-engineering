#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import yaml, numpy as np, torch
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rescontact.data.dataset import PDBContactDataset
from rescontact.api.monitor import DriftMonitor, MetricSpec

def to_np(x):
    """Return a NumPy array regardless of whether x is torch.Tensor or already NumPy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def upper_triangle(vals_2d: np.ndarray) -> np.ndarray:
    L = vals_2d.shape[0]
    iu = np.triu_indices(L, k=1)
    return vals_2d[iu]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rescontact.yaml")
    ap.add_argument("--out", default="monitor/baseline.json")
    ap.add_argument("--max_examples", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.5)  # kept for compatibility (unused here)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = PDBContactDataset(
        root_dir=cfg["paths"]["train_dir"],
        cache_dir=cfg["paths"]["cache_dir"],
        contact_threshold=cfg["labels"]["contact_threshold_angstrom"],
        include_inter_chain=cfg["labels"]["include_inter_chain"],
        esm_model_name=cfg["model"]["esm_model"],
        use_msa=bool(cfg["features"]["use_msa"]),
        msa_cfg=cfg["features"]["msa"],
    )
    n = min(len(ds), args.max_examples)

    # Collect samples for each metric we want to track
    samples = {
        "seq_len": [],
        "prob_scores": [],
        "pos_distance": [],
        "emb_norms": [],
        "msa_coverage": [],  # fraction nonzero in last-21 dims (if D>=341); else 0
    }

    for i in range(n):
        item = ds[i]
        emb  = to_np(item["emb"])         # [L, D] NumPy
        y    = to_np(item["contacts"])    # [L, L] NumPy
        mask = to_np(item["mask"])        # [L, L] NumPy
        L    = emb.shape[0]

        # (1) sequence length
        samples["seq_len"].append(np.array([L], dtype=float))

        # (2) "prob scores" baseline — use label prevalence (upper triangle of y*mask)
        y_masked_ut = upper_triangle(y * mask).astype(float)
        samples["prob_scores"].append(y_masked_ut)

        # (3) positive distances (where y==1 and mask==1)
        ii, jj = np.where((y > 0.5) & (mask > 0.5))
        if ii.size > 0:
            d = np.abs(ii - jj).astype(float)
            samples["pos_distance"].append(d)

        # (4) embedding norms per residue
        samples["emb_norms"].append(np.linalg.norm(emb, axis=1).astype(float))

        # (5) msa coverage
        if emb.shape[1] >= 341:
            last21 = emb[:, -21:]
            nz = (last21 != 0).sum()
            cov = float(nz) / float(last21.size)
        else:
            cov = 0.0
        samples["msa_coverage"].append(np.array([cov], dtype=float))

    # Concatenate lists → arrays
    samples = {k: np.concatenate(v) if len(v) else np.array([]) for k, v in samples.items()}

    specs = [
        MetricSpec("seq_len",      extractor=lambda ctx: ctx["seq_len"],      bins=10),
        MetricSpec("prob_scores",  extractor=lambda ctx: ctx["prob_scores"],  bins=10),
        MetricSpec("pos_distance", extractor=lambda ctx: ctx["pos_distance"], bins=10),
        MetricSpec("emb_norms",    extractor=lambda ctx: ctx["emb_norms"],    bins=10),
        MetricSpec("msa_coverage", extractor=lambda ctx: ctx["msa_coverage"], bins=10),
    ]
    mon = DriftMonitor(specs)
    mon.fit_baseline(samples)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mon.save_baseline(str(out_path))
    print(f"[baseline] wrote {out_path}")

if __name__ == "__main__":
    main()

