#!/usr/bin/env python3
import argparse, time, json, random, sys
from pathlib import Path
import numpy as np
import yaml, torch
from torch.utils.data import DataLoader, Subset, default_collate

# repo paths
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rescontact.data.dataset import PDBContactDataset
from scripts.train import SimpleContactNet, align_embed_dim

# ---- optional: safe collate fallback (drop None samples) ----
try:
    from scripts.train import collate_skip_none  # preferred
except Exception:
    def _sanitize_sample(sample):
        if sample is None:
            return None
        out = dict(sample)
        # drop if any core tensor missing
        for k in ("emb", "contacts", "mask"):
            if out.get(k, None) is None:
                return None
        # normalize optional fields
        if out.get("msa_path") is None:
            out["msa_path"] = ""
        return out
    def collate_skip_none(batch):
        batch = [_sanitize_sample(b) for b in batch if b is not None]
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return default_collate(batch)

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def to_np(x): 
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def upper_ut(L, device=None):
    iu = torch.triu_indices(L, L, 1, device=device)
    return iu[0], iu[1]

def hist(values: np.ndarray, bins):
    h, _ = np.histogram(values, bins=bins); s = h.sum()
    return (h / s) if s > 0 else np.zeros(len(bins)-1, dtype=float)

def psi(p_exp: np.ndarray, p_obs: np.ndarray, eps=1e-6):
    p = np.clip(p_exp, eps, 1.0); q = np.clip(p_obs, eps, 1.0)
    return float(np.sum((q - p) * np.log(q / p)))

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline", default="monitor/baseline.json")
    ap.add_argument("--split", choices=["train","val"], default="val")
    ap.add_argument("--max-examples", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    # accept both singular and plural; both map to the same dest
    ap.add_argument("--use-checkpoint", dest="use_checkpoint", action="store_true")
    ap.add_argument("--use-checkpoints", dest="use_checkpoint", action="store_true")
    ap.add_argument("--outdir", default="monitor/reports/psi_run")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load baseline (bins + expected hists)
    base = json.load(open(args.baseline))
    score_bins = np.array(base["bins"]["score"], dtype=float)
    sep_bins   = np.array(base["bins"]["sep"], dtype=float)
    len_bins   = np.array(base["bins"]["length"], dtype=float)

    # Dataset & split identical to training
    ds_full = PDBContactDataset(
        root_dir=cfg["paths"]["train_dir"],
        cache_dir=cfg["paths"]["cache_dir"],
        contact_threshold=cfg["labels"]["contact_threshold_angstrom"],
        include_inter_chain=cfg["labels"]["include_inter_chain"],
        esm_model_name=cfg["model"]["esm_model"],
        use_msa=bool(cfg["features"]["use_msa"]),
        msa_cfg=cfg["features"]["msa"],
    )
    idx = list(range(len(ds_full)))
    random.Random(cfg["project"].get("seed", args.seed)).shuffle(idx)
    frac = float(cfg["training"]["train_val_split"])
    n_tr = max(1, int(len(ds_full) * frac))
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
    use_idx = tr_idx if args.split == "train" else va_idx
    use_idx = use_idx[:args.max_examples]
    ds = Subset(ds_full, use_idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_skip_none)

    # Device & model for predicted probs
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
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
            print(f"[psi] loaded checkpoint: {ckpt}")
        else:
            print(f"[psi] WARN: {ckpt} not found; falling back to label prevalence baseline.")
            model = None

    # Collect current distributions
    cur_scores, cur_seps, cur_lengths = [], [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None: 
                continue
            emb = batch["emb"].to(device).float()
            y   = batch["contacts"].to(device).float()
            m   = batch["mask"].to(device).float()
            emb = align_embed_dim(emb, want_dim)
            B, L, _ = emb.shape
            i_ut, j_ut = upper_ut(L, emb.device)
            cur_lengths.extend([L] * B)

            if model is not None:
                logits = model(emb); probs = torch.sigmoid(logits)
                for b in range(B):
                    mb = m[b][i_ut, j_ut] > 0.5
                    if mb.sum().item() == 0: 
                        continue
                    pv = probs[b][i_ut, j_ut][mb].detach().cpu().numpy()
                    cur_scores.append(pv)
                    yy = (y[b][i_ut, j_ut][mb] > 0.5)
                    if yy.any():
                        cur_seps.append((j_ut[mb][yy] - i_ut[mb][yy]).detach().cpu().numpy())
            else:
                # label-prevalence fallback
                for b in range(B):
                    mb = m[b][i_ut, j_ut] > 0.5
                    if mb.sum().item() == 0: continue
                    yv = (y[b][i_ut, j_ut][mb] > 0.5).detach().cpu().numpy().astype(np.float32)
                    cur_scores.append(yv)
                    if yv.any():
                        cur_seps.append((j_ut[mb][yv > 0] - i_ut[mb][yv > 0]).detach().cpu().numpy())

    def cat(lst):
        return np.concatenate(lst) if len(lst) else np.array([], dtype=float)

    scores = cat(cur_scores)
    seps   = cat(cur_seps)
    lens   = np.array(cur_lengths, dtype=float)
    if scores.size == 0:
        raise SystemExit("No valid pairs to score—check masks/data.")

    # build current hists
    cur = {
        "score":  hist(scores, score_bins),
        "sep":    hist(seps,   sep_bins),
        "length": hist(lens,   len_bins),
    }
    exp = {
        "score":  np.array(base["expected"]["score"], dtype=float),
        "sep":    np.array(base["expected"]["sep"],   dtype=float),
        "length": np.array(base["expected"]["length"],dtype=float),
    }

    # PSI
    psi_res = {
        "score":  psi(exp["score"],  cur["score"]),
        "sep":    psi(exp["sep"],    cur["sep"]),
        "length": psi(exp["length"], cur["length"]),
    }

    # Save JSON + small Markdown summary
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_json = Path(outdir) / f"psi_{args.split}_{ts}.json"
    out_md   = Path(outdir) / f"psi_{args.split}_{ts}.md"

    save_json({
        "config": Path(args.config).name,
        "split": args.split,
        "max_examples": args.max_examples,
        "use_checkpoint": bool(model is not None),
        "psi": psi_res,
        "bins": {"score": score_bins.tolist(), "sep": sep_bins.tolist(), "length": len_bins.tolist()},
        "current": {k: v.tolist() for k, v in cur.items()},
        "expected": {k: exp[k].tolist() for k in exp},
    }, out_json)

    # Plots
    import matplotlib.pyplot as plt
    def plot_two(name, bins, expv, curv):
        centers = (bins[:-1] + bins[1:]) / 2.0
        plt.figure()
        plt.plot(centers, expv, label="baseline")
        plt.plot(centers, curv, label="current")
        plt.title(f"{name} distribution")
        plt.xlabel(name); plt.ylabel("proportion"); plt.legend()
        p = Path(outdir) / f"{name}_{args.split}_{ts}.png"
        plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
        return p

    p_score  = plot_two("score",  score_bins,  exp["score"],  cur["score"])
    p_sep    = plot_two("sep",    sep_bins,    exp["sep"],    cur["sep"])
    p_length = plot_two("length", len_bins,    exp["length"], cur["length"])

    md = []
    md.append(f"# PSI report ({args.split}) — {ts}")
    md.append(f"- Config: `{Path(args.config).name}`  \n- Examples: {args.max_examples}  \n- Predicted probs: {bool(model is not None)}")
    md.append("")
    md.append("| Metric | PSI | Interpretation |")
    md.append("|---|---:|---|")
    def interp(x):
        return "no drift (<0.1)" if x < 0.1 else ("monitor (0.1–0.25)" if x < 0.25 else "investigate (>0.25)")
    for k in ["score","sep","length"]:
        md.append(f"| {k} | {psi_res[k]:.3f} | {interp(psi_res[k])} |")
    md.append("")
    md.append(f"![score]({p_score.name})")
    md.append(f"![sep]({p_sep.name})")
    md.append(f"![length]({p_length.name})")
    out_md.write_text("\n".join(md))
    print(f"[psi] wrote {out_json}")
    print(f"[psi] wrote {out_md}")
    print(f"[psi] figs: {p_score.name}, {p_sep.name}, {p_length.name}")

if __name__ == "__main__":
    main()

