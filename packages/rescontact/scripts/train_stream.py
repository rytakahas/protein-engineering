#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming trainer for Res-Contact (ESM-only, laptop-friendly).
- Parses PDB/mmCIF on the fly (first model, first chain)
- Builds Cα–Cα contact labels (< threshold Å)
- Embeds with tiny ESM2 (cached on disk via ESMEmbedder)
- Trains a simple bilinear head (works with server.py)
- Saves checkpoint you can serve immediately

Usage:
  PYTHONPATH=src python scripts/train_stream.py \
      --config configs/rescontact.yaml \
      --epochs 3 --neg-ratio 3 --max-train 50
"""
import argparse, math, os, random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

import yaml
from Bio.PDB import MMCIFParser, PDBParser

from rescontact.features.esm import ESMEmbedder   # uses on-disk cache

# -----------------------
# Small utils
# -----------------------
AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
}

def pick_device(pref: List[str]) -> torch.device:
    for p in pref:
        if p == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
        if p == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
        if p == "cpu": return torch.device("cpu")
    return torch.device("cpu")

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def collect_structs(dirpath: Path) -> List[Path]:
    exts = (".pdb", ".cif", ".mmcif")
    files = [p for p in sorted(dirpath.rglob("*")) if p.suffix.lower() in exts]
    return files

def extract_seq_ca(path: Path) -> Tuple[str, np.ndarray]:
    """
    Return (sequence, CA_coords[L,3]) from first model/chain.
    Non-standard residues or those without CA are skipped.
    """
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in (".cif",".mmcif") else PDBParser(QUIET=True)
    s = parser.get_structure("X", str(path))
    model = next(s.get_models())
    chain = next(model.get_chains())

    seq = []
    cas  = []
    for res in chain.get_residues():
        name = res.get_resname().upper()
        if name in AA3_TO_1 and "CA" in res:
            seq.append(AA3_TO_1[name])
            cas.append(res["CA"].get_coord().astype(np.float32))
    if not seq or not cas:
        raise ValueError(f"No valid residues with CA in {path.name}")
    return "".join(seq), np.stack(cas, axis=0)  # [L,3]

def contact_map_from_ca(ca: np.ndarray, thresh: float = 8.0) -> np.ndarray:
    """Boolean L×L contacts (no diagonal) from CA coords and threshold in Å."""
    L = ca.shape[0]
    diffs = ca[:, None, :] - ca[None, :, :]
    d2 = np.sum(diffs*diffs, axis=-1)
    dist = np.sqrt(d2, dtype=np.float32)
    cm = (dist < thresh).astype(np.uint8)
    np.fill_diagonal(cm, 0)
    return cm

def upper_pairs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (i,j) indices for upper triangle (k=1)."""
    L = mask.shape[0]
    iu = np.triu_indices(L, k=1)
    return iu[0], iu[1]

# -----------------------
# Simple contact model (matches server fallback)
# -----------------------
class SimpleContactNet(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, distance_bias_max: int = 512):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.bilin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dist_bias = nn.Embedding(distance_bias_max, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: [L,D] or [1,L,D]
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        z  = self.act(self.proj(emb))                    # [B,L,H]
        zW = self.bilin(z)                               # [B,L,H]
        logits = torch.einsum("blh,bmh->blm", zW, z) / math.sqrt(z.shape[-1])  # [B,L,L]
        B, L, _ = logits.shape
        idx = torch.arange(L, device=logits.device)
        dist = (idx[None, :] - idx[:, None]).abs().clamp_max(self.dist_bias.num_embeddings - 1)
        db = self.dist_bias(dist)[:, :, 0]               # [L,L]
        return logits + db.unsqueeze(0)                  # [B,L,L]

# -----------------------
# Training / Eval (streaming)
# -----------------------
def sample_pairs(contact_u: np.ndarray, neg_ratio: int = 3, max_pairs: int = 20_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    contact_u: upper-tri (L,L) boolean matrix
    Returns i,j,y (y∈{0,1}) vectors for sampled pairs.
    """
    L = contact_u.shape[0]
    iu, ju = upper_pairs(np.ones((L,L), dtype=bool))
    mask_pos = contact_u[iu, ju] == 1
    pos_i, pos_j = iu[mask_pos], ju[mask_pos]

    # Negatives from upper-tri where label==0
    neg_mask = ~mask_pos
    neg_i_all, neg_j_all = iu[neg_mask], ju[neg_mask]
    n_pos = len(pos_i)
    n_neg = min(len(neg_i_all), n_pos * max(1, neg_ratio))

    if n_neg > 0:
        sel = np.random.choice(len(neg_i_all), size=n_neg, replace=False)
        neg_i, neg_j = neg_i_all[sel], neg_j_all[sel]
    else:
        neg_i, neg_j = np.array([], dtype=int), np.array([], dtype=int)

    i = np.concatenate([pos_i, neg_i])
    j = np.concatenate([pos_j, neg_j])
    y = np.concatenate([np.ones_like(pos_i), np.zeros_like(neg_i)]).astype(np.float32)

    # Cap total pairs
    if max_pairs and len(i) > max_pairs:
        sel = np.random.choice(len(i), size=max_pairs, replace=False)
        i, j, y = i[sel], j[sel], y[sel]
    return i, j, y

@torch.no_grad()
def evaluate_files(files: List[Path], embedder: ESMEmbedder, model: nn.Module, device: torch.device,
                   thresh: float, max_pairs_per_struct: int = 80_000) -> Dict[str, float]:
    y_true_all, y_prob_all = [], []
    for p in files:
        try:
            seq, ca = extract_seq_ca(p)
            if len(seq) < 8:
                continue
            contacts = contact_map_from_ca(ca, thresh=thresh).astype(bool)
            L = len(seq)
            # pairs from upper triangle
            iu, ju = upper_pairs(contacts)
            if len(iu) == 0:
                continue
            # cap pairs
            if max_pairs_per_struct and len(iu) > max_pairs_per_struct:
                sel = np.random.choice(len(iu), size=max_pairs_per_struct, replace=False)
                iu, ju = iu[sel], ju[sel]

            emb = torch.from_numpy(embedder.embed(seq)).float().to(device)   # [L,D]
            logits = model(emb).squeeze(0)                                   # [L,L]
            probs  = torch.sigmoid(logits).detach().cpu().numpy()

            y_true = contacts[iu, ju].astype(np.float32)
            y_prob = probs[iu, ju].astype(np.float32)

            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
        except Exception:
            continue
    if not y_true_all:
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}
    y_true_all = np.concatenate(y_true_all)
    y_prob_all = np.concatenate(y_prob_all)
    pr = average_precision_score(y_true_all, y_prob_all)
    try:
        roc = roc_auc_score(y_true_all, y_prob_all)
    except ValueError:
        roc = float("nan")
    return {"pr_auc": float(pr), "roc_auc": float(roc)}

def train_stream(files: List[Path], val_files: List[Path], cfg: dict):
    device = pick_device(cfg["project"].get("device_preference", ["mps","cpu"]))
    set_seed(int(cfg["project"].get("seed", 42)))
    cache_dir = Path(cfg["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    thresh = float(cfg["labels"]["contact_threshold_angstrom"])
    embed_dim = int(cfg["model"]["embed_dim"])
    hidden_dim = int(cfg["model"]["hidden_dim"])
    dist_bias_max = int(cfg["model"]["distance_bias_max"])
    esm_name = cfg["model"]["esm_model"]

    epochs = int(cfg["training"].get("epochs", 3))
    lr     = float(cfg["training"].get("lr", 1.5e-3))
    neg_ratio = int(cfg.get("streaming", {}).get("neg_ratio", 3))
    train_batch_pairs = int(cfg.get("streaming", {}).get("train_batch_pairs", 20_000))

    embedder = ESMEmbedder(esm_name, cache_dir=str(cache_dir/"emb"), device=device)
    model = SimpleContactNet(embed_dim=embed_dim, hidden_dim=hidden_dim, distance_bias_max=dist_bias_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(1, epochs+1):
        total_loss, seen = 0.0, 0
        random.shuffle(files)
        for p in files:
            try:
                seq, ca = extract_seq_ca(p)
                if len(seq) < 8:  # skip tiny chains
                    continue
                contacts = contact_map_from_ca(ca, thresh=thresh).astype(bool)
                i, j, y = sample_pairs(contacts, neg_ratio=neg_ratio, max_pairs=train_batch_pairs)

                emb = torch.from_numpy(embedder.embed(seq)).float().to(device)   # [L,D]
                logits = model(emb).squeeze(0)                                   # [L,L]
                y_hat = logits[i, j]                                             # [P]
                y_t   = torch.from_numpy(y).to(device)

                loss = loss_fn(y_hat, y_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1.0)))
                opt.step()

                total_loss += float(loss.item()); seen += 1
            except Exception:
                continue
        avg_loss = total_loss / max(seen, 1)

        # quick val
        model.eval()
        val_scores = evaluate_files(val_files, embedder, model, device, thresh)
        model.train()

        print(f"[epoch {ep}] loss={avg_loss:.4f}  val PR-AUC={val_scores['pr_auc']:.4f}  ROC-AUC={val_scores['roc_auc']:.4f}")

    # save
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / "rescontact_stream.pt"
    torch.save({"state_dict": model.state_dict()}, out)
    print(f"[save] {out.resolve()}")
    return out

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rescontact.yaml")
    ap.add_argument("--max-train", type=int, default=0, help="cap number of train files")
    ap.add_argument("--max-val", type=int, default=0, help="cap number of val files")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--neg-ratio", type=int, default=None, help="override neg ratio")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    # allow small overrides from CLI
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.neg_ratio is not None:
        cfg.setdefault("streaming", {})["neg_ratio"] = args.neg_ratio

    train_dir = Path(cfg["paths"]["train_dir"])
    files = collect_structs(train_dir)
    if args.max_train and len(files) > args.max_train:
        files = files[: args.max_train]

    # split 80/20
    tr, val = train_test_split(files, test_size=0.2, random_state=int(cfg["project"].get("seed", 42)))
    if args.max_val and len(val) > args.max_val:
        val = val[: args.max_val]

    ckpt_path = train_stream(tr, val, cfg)

    # Optionally copy to model_best.pt for server compatibility
    best = Path(cfg["paths"]["ckpt_dir"]) / "model_best.pt"
    try:
        import shutil
        shutil.copy2(ckpt_path, best)
        print(f"[link] copied to {best.resolve()}  (server will pick this up)")
    except Exception:
        pass

if __name__ == "__main__":
    main()

