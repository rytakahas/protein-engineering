#!/usr/bin/env python3
import os, math, random, time, platform, argparse, contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate, Subset

# Optional metrics
try:
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False
    import numpy as np


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device(pref: List[str]) -> torch.device:
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
    # Ensure no Nones inside the dict (msa_path is allowed to be empty string)
    if out.get("msa_path") is None:
        out["msa_path"] = ""
    # emb/contacts/mask must be tensors; if any is missing, drop this sample
    for k in ("emb", "contacts", "mask"):
        if out.get(k, None) is None:
            return None
    return out

def collate_skip_none(batch: List[Optional[Dict[str, Any]]]):
    batch = [sanitize_sample(b) for b in batch if b is not None]
    batch = [b for b in batch if b is not None]
    if not batch:
        return None  # training/eval loops skip it
    return default_collate(batch)

def align_embed_dim(x: torch.Tensor, want: int) -> torch.Tensor:
    """Match last-dim to 'want': pad zeros if smaller, slice if larger."""
    got = x.shape[-1]
    if got == want:
        return x
    if got < want:
        pad = want - got
        pad_t = x.new_zeros(*x.shape[:-1], pad)
        return torch.cat([x, pad_t], dim=-1)
    return x[..., :want]


# -----------------------
# Very small contact model
# -----------------------
class SimpleContactNet(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dist_bias_max: int = 512, dropout_p: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_p)  # <-- added dropout (default 0.1)
        self.bilin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dist_bias = nn.Embedding(dist_bias_max, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: [B, L, D]
        z = self.drop(self.act(self.proj(emb)))  # [B,L,H] + dropout
        zW = self.bilin(z)
        logits = torch.einsum("blh,bmh->blm", zW, z) / math.sqrt(z.shape[-1])
        B, L, _ = logits.shape
        idx = torch.arange(L, device=logits.device)
        dist = (idx[None, :] - idx[:, None]).abs().clamp_max(self.dist_bias.num_embeddings - 1)
        db = self.dist_bias(dist)[:, :, 0]  # [L,L]
        logits = logits + db.unsqueeze(0)   # broadcast over batch
        return logits  # [B,L,L]


# -----------------------
# Loss / metrics
# -----------------------
def bce_loss_with_mask(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom

def _upper_flat(arr: torch.Tensor) -> torch.Tensor:
    """Flatten strictly upper-triangular part (i<j)."""
    L = arr.shape[-1]
   # MPS doesn’t implement triu_indices; create on CPU, then move.
    iu = torch.triu_indices(L, L, offset=1)  # CPU by default
    if iu.device != arr.device:
        iu = iu.to(arr.device)
    return arr[..., iu[0], iu[1]]
   # iu = torch.triu_indices(L, L, offset=1, device=arr.device)
   # return arr[..., iu[0], iu[1]]

def _batch_metrics(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> Dict[str, float]:
    B, L, _ = y.shape
    probs = torch.sigmoid(logits)

    pals: List[float] = []
    rocs: List[float] = []
    prs:  List[float] = []
    f1s:  List[float] = []

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
            topk = np.argpartition(-pv, kth=k-1)[:k]
            pals.append(float(yv[topk].mean()))

        if HAVE_SK:
            try:
                rocs.append(float(roc_auc_score(yv, pv)))
            except Exception:
                pass
            try:
                prs.append(float(average_precision_score(yv, pv)))
            except Exception:
                pass

        pred = (pv >= 0.5).astype(np.int32)
        tp = int((pred == 1).astype(np.int32)[yv == 1].sum())
        fp = int((pred == 1).astype(np.int32)[yv == 0].sum())
        fn = int((pred == 0).astype(np.int32)[yv == 1].sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(float(f1))

    def _avg(xs): return float(np.mean(xs)) if len(xs) else float("nan")
    return {"pal": _avg(pals), "roc": _avg(rocs), "pr": _avg(prs), "f1": _avg(f1s)}


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, want_dim: int) -> Tuple[float, Dict[str, float]]:
    model.eval()
    tot, n = 0.0, 0
    pal_list, roc_list, pr_list, f1_list = [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        emb = batch["emb"].to(device).float()
        y   = batch["contacts"].to(device).float()
        m   = batch["mask"].to(device).float()
        emb = align_embed_dim(emb, want_dim)
        logits = model(emb)
        loss = bce_loss_with_mask(logits, y, m)
        tot += float(loss.item()); n += 1

        met = _batch_metrics(logits, y, m)
        if not math.isnan(met["pal"]): pal_list.append(met["pal"])
        if not math.isnan(met["roc"]): roc_list.append(met["roc"])
        if not math.isnan(met["pr"]):  pr_list.append(met["pr"])
        if not math.isnan(met["f1"]):  f1_list.append(met["f1"])

    def _avg(xs): return float(np.mean(xs)) if len(xs) else float("nan")
    metrics = {"pal": _avg(pal_list), "roc": _avg(roc_list), "pr": _avg(pr_list), "f1": _avg(f1_list)}
    return tot / max(n, 1), metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                    optimizer: torch.optim.Optimizer, use_amp: bool, want_dim: int,
                    max_batches: Optional[int] = None, debug: bool = False) -> float:
    model.train()
    tot, n = 0.0, 0
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else contextlib.nullcontext
    with amp_ctx():
        for bi, batch in enumerate(loader, start=1):
            if batch is None:
                if debug: print("[train] skipped None batch")
                continue
            emb = batch["emb"].to(device).float()
            y   = batch["contacts"].to(device).float()
            m   = batch["mask"].to(device).float()
            emb = align_embed_dim(emb, want_dim)

            optimizer.zero_grad(set_to_none=True)
            logits = model(emb)
            loss = bce_loss_with_mask(logits, y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot += float(loss.item()); n += 1
            if max_batches and bi >= max_batches:
                break
    return tot / max(n, 1)


# -----------------------
# CLI
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train simple contact predictor (ESM(+MSA)).")
    ap.add_argument("--config", default="configs/rescontact.yaml", help="Path to YAML config.")
    ap.add_argument("--max_train_examples", type=int, default=None, help="Override limits.train_max_examples.")
    ap.add_argument("--subset_mode", choices=["random", "first"], default=None, help="Override limits.subset_mode.")
    ap.add_argument("--train_val_split", type=float, default=None, help="Override training.train_val_split (e.g., 0.8).")
    ap.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    ap.add_argument("--batch_size", type=int, default=None, help="Override training.batch_size.")
    ap.add_argument("--debug", action="store_true", help="Extra logs; used with --max_train_batches for smoke tests.")
    ap.add_argument("--max_train_batches", type=int, default=None, help="Iterate only N batches per epoch.")
    # The following CLI MSA flags are *optional* and do nothing unless you pass them:
    ap.add_argument("--msa", action="store_true", help="Force-enable MSA (overrides YAML).")
    ap.add_argument("--no_msa", action="store_true", help="Force-disable MSA (overrides YAML).")
    ap.add_argument("--embed_dim", type=int, help="Override model.embed_dim (320 or 341).")
    return ap.parse_args()


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Apply CLI overrides
    if args.max_train_examples is not None:
        cfg["limits"]["train_max_examples"] = int(args.max_train_examples)
    if args.subset_mode is not None:
        cfg["limits"]["subset_mode"] = args.subset_mode
    if args.train_val_split is not None:
        cfg["training"]["train_val_split"] = float(args.train_val_split)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)

    if args.msa:
        cfg["features"]["use_msa"] = True
    if args.no_msa:
        cfg["features"]["use_msa"] = False
    if args.embed_dim is not None:
        cfg["model"]["embed_dim"] = int(args.embed_dim)

    set_seed(int(cfg["project"]["seed"]))
    device = pick_device(cfg["project"].get("device_preference", ["cuda", "mps", "cpu"]))

    print(f"[rescontact] loading config from {args.config}")
    print(f"[rescontact] torch={torch.__version__}  python={platform.python_version()}")
    print(f"[rescontact] Using device: {device.type}")

    # ---- Dataset ----
    from rescontact.data.dataset import PDBContactDataset
    ds_full = PDBContactDataset(
        root_dir=cfg["paths"]["train_dir"],
        cache_dir=cfg["paths"]["cache_dir"],
        contact_threshold=cfg["labels"]["contact_threshold_angstrom"],
        include_inter_chain=cfg["labels"]["include_inter_chain"],
        esm_model_name=cfg["model"]["esm_model"],
        use_msa=bool(cfg["features"]["use_msa"]),
        msa_cfg=cfg["features"]["msa"],
    )

    # Optional tiny subset
    lim = int(cfg["limits"]["train_max_examples"])
    mode = cfg["limits"].get("subset_mode", "random")
    idx = list(range(len(ds_full)))
    if lim and lim < len(idx):
        if mode == "random":
            random.Random(cfg["project"]["seed"]).shuffle(idx)
        idx = idx[:lim]
        ds_full = Subset(ds_full, idx)

    # Train/Val split
    frac = float(cfg["training"]["train_val_split"])
    n_tr = max(1, int(len(ds_full) * frac))
    n_va = max(1, len(ds_full) - n_tr)
    tr_idx = list(range(n_tr))
    va_idx = list(range(n_tr, n_tr + n_va))
    ds_tr = Subset(ds_full, tr_idx)
    ds_va = Subset(ds_full, va_idx)

    print(f"[rescontact] dataset ready: total={len(ds_full)}  train={len(ds_tr)}  val={len(ds_va)}")

    # Warm-up one item to trigger ESM cache (non-fatal if it fails)
    try:
        _ = ds_tr[0]
        print("[rescontact/ESM] init model_id=facebook/esm2_t6_8M_UR50D cache=.cache/rescontact/emb device=mps")
        print("[rescontact] debug: first item ready")
    except Exception as e:
        print(f"[rescontact] warm-up failed: {e}")

    # ---- Model/Opt ----
    want_dim = int(cfg["model"]["embed_dim"])   # 320 (no MSA) or 341 (with MSA-1D)
    model = SimpleContactNet(
        embed_dim=want_dim,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dist_bias_max=int(cfg["model"]["distance_bias_max"]),
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[rescontact] model built ({nparams:.2f}M params) → start training")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    use_amp = bool(cfg["training"]["mixed_precision"]) and (device.type == "cuda")

    # ---- Loaders ----
    bs = int(cfg["training"]["batch_size"])
    loader_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0, collate_fn=collate_skip_none)
    loader_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_skip_none)

    # ---- Train ----
    epochs = int(cfg["training"]["epochs"])
    best_val = float("inf")
    patience = int(cfg["training"]["patience"])
    bad = 0
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, loader_tr, device, optimizer, use_amp, want_dim,
            max_batches=args.max_train_batches, debug=args.debug
        )
        va_loss, va_metrics = eval_epoch(model, loader_va, device, want_dim)
        dt = time.time() - t0

        roc = va_metrics["roc"]; pr = va_metrics["pr"]; pal = va_metrics["pal"]; f1 = va_metrics["f1"]
        roc_s = f"{roc:.3f}" if not math.isnan(roc) else "na"
        pr_s  = f"{pr:.3f}"  if not math.isnan(pr)  else "na"
        pal_s = f"{pal:.3f}" if not math.isnan(pal) else "na"
        f1_s  = f"{f1:.3f}"  if not math.isnan(f1)  else "na"
        print(f"[epoch {ep}] train={tr_loss:.4f}  val={va_loss:.4f}  P@L={pal_s}  ROC={roc_s}  PR={pr_s}  F1={f1_s}  ({dt:.1f}s)")

        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            bad = 0
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir / "model_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("[rescontact] Early stopping.")
                break


if __name__ == "__main__":
    main()
