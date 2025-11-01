#!/usr/bin/env python3
import math, platform, argparse, random, contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate, Subset

try:
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False
    import numpy as np


def pick_device(pref):
    for p in pref:
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if p == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")

def sanitize_sample(sample: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if sample is None:
        return None
    out = dict(sample)
    if out.get("msa_path") is None:
        out["msa_path"] = ""
    return out

def collate_skip_none(batch: List[Optional[Dict[str, Any]]]):
    batch = [sanitize_sample(b) for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

def align_embed_dim(x: torch.Tensor, want: int) -> torch.Tensor:
    got = x.shape[-1]
    if got == want:
        return x
    if got < want:
        pad = want - got
        return torch.cat([x, x.new_zeros(*x.shape[:-1], pad)], dim=-1)
    return x[..., :want]

class SimpleContactNet(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dist_bias_max: int = 512):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.bilin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dist_bias = nn.Embedding(dist_bias_max, 1)
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        z = self.act(self.proj(emb))
        zW = self.bilin(z)
        logits = torch.einsum("blh,bmh->blm", zW, z) / math.sqrt(z.shape[-1])
        B, L, _ = logits.shape
        idx = torch.arange(L, device=logits.device)
        dist = (idx[None, :] - idx[:, None]).abs().clamp_max(self.dist_bias.num_embeddings - 1)
        db = self.dist_bias(dist)[:, :, 0]
        return logits + db.unsqueeze(0)

def _upper_flat(arr: torch.Tensor) -> torch.Tensor:
    L = arr.shape[-1]
    # MPS doesn't implement triu_indices → create on CPU, then move.
    iu = torch.triu_indices(L, L, offset=1)  # CPU by default
    if iu.device != arr.device:
        iu = iu.to(arr.device)
    return arr[..., iu[0], iu[1]]

@torch.no_grad()
def evaluate_split(ds, device, model, want_dim: int, batch_size: int = 1) -> Dict[str, float]:
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_skip_none)
    model.eval()
    tot, n = 0.0, 0
    pals, rocs, prs, f1s = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        emb = batch["emb"].to(device).float()
        y   = batch["contacts"].to(device).float()
        m   = batch["mask"].to(device).float()
        emb = align_embed_dim(emb, want_dim)
        logits = model(emb)

        # loss
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * m).sum() / m.sum().clamp(min=1.0)
        tot += float(loss.item()); n += 1

        # metrics
        probs = torch.sigmoid(logits)
        B, L, _ = y.shape
        for b in range(B):
            yb = _upper_flat(y[b]).detach().cpu()
            pb = _upper_flat(probs[b]).detach().cpu()
            mb = _upper_flat(m[b]).detach().cpu().bool()
            if mb.sum().item() == 0:
                continue
            yv = yb[mb].numpy()
            pv = pb[mb].numpy()
            # P@L
            k = min(L, len(pv))
            if k > 0:
                idx = np.argpartition(-pv, kth=k-1)[:k]
                pals.append(float(yv[idx].mean()))
            # ROC / PR
            if HAVE_SK:
                try:
                    rocs.append(float(roc_auc_score(yv, pv)))
                except Exception:
                    pass
                try:
                    prs.append(float(average_precision_score(yv, pv)))
                except Exception:
                    pass
            # F1@0.5
            pred = (pv >= 0.5).astype(np.int32)
            tp = int((pred == 1).astype(np.int32)[yv == 1].sum())
            fp = int((pred == 1).astype(np.int32)[yv == 0].sum())
            fn = int((pred == 0).astype(np.int32)[yv == 1].sum())
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            f1s.append(float(f1))

    def _avg(xs): return float(np.mean(xs)) if len(xs) else float("nan")
    return {
        "loss": tot / max(n, 1),
        "pal": _avg(pals),
        "roc": _avg(rocs),
        "pr":  _avg(prs),
        "f1":  _avg(f1s),
    }

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate contact model.")
    ap.add_argument("--config", default="configs/rescontact.yaml", help="Path to YAML config.")
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint (default: checkpoints/model_best.pt).")
    ap.add_argument("--split", choices=["test", "train"], default="test", help="Which directory to evaluate on.")
    ap.add_argument("--max_test_examples", type=int, default=None, help="Limit #structures in this eval run.")
    ap.add_argument("--subset_mode", choices=["random", "first"], default=None, help="Subset mode if limiting.")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = pick_device(cfg["project"].get("device_preference", ["cuda", "mps", "cpu"]))
    print(f"[rescontact/eval] torch={torch.__version__}  python={platform.python_version()}  device={device.type}")

    want_dim = int(cfg["model"]["embed_dim"])
    from rescontact.data.dataset import PDBContactDataset

    data_dir = cfg["paths"]["test_dir"] if args.split == "test" else cfg["paths"]["train_dir"]
    ds_full = PDBContactDataset(
        root_dir=data_dir,
        cache_dir=cfg["paths"]["cache_dir"],
        contact_threshold=cfg["labels"]["contact_threshold_angstrom"],
        include_inter_chain=cfg["labels"]["include_inter_chain"],
        esm_model_name=cfg["model"]["esm_model"],
        use_msa=bool(cfg["features"]["use_msa"]),
        msa_cfg=cfg["features"]["msa"],
    )

    # Limit examples (CLI first, YAML fallback)
    lim = args.max_test_examples if args.max_test_examples is not None else int(cfg["limits"].get("test_max_examples", 0) or 0)
    if lim and lim < len(ds_full):
        idx = list(range(len(ds_full)))
        mode = args.subset_mode or cfg["limits"].get("subset_mode", "random")
        if mode == "random":
            random.Random(cfg["project"]["seed"]).shuffle(idx)
        idx = idx[:lim]
        ds_full = Subset(ds_full, idx)

    print(f"[rescontact/eval] evaluating split={args.split}  size={len(ds_full)}")

    model = SimpleContactNet(
        embed_dim=want_dim,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dist_bias_max=int(cfg["model"]["distance_bias_max"]),
    ).to(device)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg["paths"]["ckpt_dir"]) / "model_best.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    print(f"[rescontact/eval] loaded checkpoint: {ckpt_path}")

    metrics = evaluate_split(ds_full, device, model, want_dim, batch_size=int(cfg["training"]["batch_size"]))
    roc_s = f"{metrics['roc']:.3f}" if not math.isnan(metrics['roc']) else "na"
    pr_s  = f"{metrics['pr']:.3f}"  if not math.isnan(metrics['pr'])  else "na"
    pal_s = f"{metrics['pal']:.3f}" if not math.isnan(metrics['pal']) else "na"
    f1_s  = f"{metrics['f1']:.3f}"  if not math.isnan(metrics['f1'])  else "na"
    #print(f"[rescontact/eval] loss={metrics['loss']:.4f}  P@L={pal_s}  ROC={roc_s}  PR={pr_s}  F1={f1_s}")
    print(f"[rescontact/eval] loss={metrics['loss']:.4f}  ROC={roc_s}  PR={pr_s}  F1={f1_s}")

if __name__ == "__main__":
    main()

