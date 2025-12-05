#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ResIntNet memory surrogate with an 80/20 split and report metrics.
Optionally fine-tune on per-variant DMS CSV (dense supervision), run edge-ablation
sanity checks and RL refinement of g_mem.

Examples
--------
# Basic (graph-level weak labels, proxy influence on val)
python scripts/train_memory.py \
  --terms TP53 LYSOZYME \
  --n_structures 20 \
  --train_steps 200 \
  --out ./outputs

# With per-variant fine-tune on a single protein's DMS CSV
python scripts/train_memory.py \
  --terms TEM1 \
  --n_structures 30 \
  --train_steps 200 \
  --dms_csv ./data/tem1_dms.csv \
  --dms_eval_frac 0.2 \
  --dms_steps 1500 \
  --out ./outputs_tem1

Notes
-----
- Graph-level targets are weak (per-protein means); use --dms_csv for real variant-level evaluation.
- Correlations are skipped when sample size/variance is insufficient.
"""

import os, json, argparse, random
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Core package (your repo) ----
from resintnet.memory_flow import (
    collect_uniprot_via_terms,
    uniprot_to_structures,
    build_graphs_from_structures,
    fit_feature_scalers,
    train_graph_regressor_amp_route,
    predict_graph_amp,
    graph_targets_for_uniprot,
    validate_influence_edges_sweep,
    ablate_edges_by_rank,
    rl_refine_gmem,
    clone_graph,
    laplacian_pinv_weighted,
    prs_edge_flux,
    adapt_conductance,
    set_node_aa,
    to_features_device,
)

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def split_indices(n: int, val_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return sorted(train_idx.tolist()), sorted(val_idx.tolist())

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    if np.isclose(x.std(), 0) or np.isclose(y.std(), 0):
        return np.nan, np.nan
    xm, ym = x - x.mean(), y - y.mean()
    pear = float((xm * ym).sum() / (np.sqrt((xm**2).sum() * (ym**2).sum()) + 1e-12))
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rxm, rym = rx - rx.mean(), ry - ry.mean()
    spear = float((rxm * rym).sum() / (np.sqrt((rxm**2).sum() * (rym**2).sum()) + 1e-12))
    return pear, spear

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2)) if len(y_true) else np.nan
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else np.nan
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan
    if len(y_true) >= 2 and not np.isclose(np.var(y_true), 0.0):
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan
    pear, spear = safe_corr(y_true, y_pred)
    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
        "Pearson": pear, "Spearman": spear, "N": int(len(y_true)),
    }

def evaluate_split(name, graphs, scalers, model, out_dir: Path):
    rows = []
    for (acc, pdbid, chain, G) in graphs:
        A_t, ddG_t = graph_targets_for_uniprot(acc)  # weak per-protein means
        A_hat, ddG_hat = predict_graph_amp(model, G, scalers)
        rows.append({
            "uniprot": acc, "pdb_id": pdbid, "chain": chain,
            "A_target": A_t, "ddG_target": ddG_t,
            "A_hat": A_hat, "ddG_hat": ddG_hat,
            "err_A": A_hat - A_t, "err_ddG": ddG_hat - ddG_t,
        })
    df = pd.DataFrame(rows)
    df_path = out_dir / f"predictions_{name}.csv"
    df.to_csv(df_path, index=False)

    m_A   = regression_metrics(df["A_target"].values,   df["A_hat"].values)
    m_ddG = regression_metrics(df["ddG_target"].values, df["ddG_hat"].values)
    mse_both  = float(np.mean(((df["A_hat"]-df["A_target"])**2 + (df["ddG_hat"]-df["ddG_target"])**2)/2.0)) if len(df) else np.nan
    rmse_both = float(np.sqrt(mse_both)) if np.isfinite(mse_both) else np.nan
    mae_both  = float(np.mean((np.abs(df["A_hat"]-df["A_target"]) + np.abs(df["ddG_hat"]-df["ddG_target"]))/2.0)) if len(df) else np.nan

    metrics = {
        "split": name,
        "A": m_A, "ddG": m_ddG,
        "combined": {"MSE": mse_both, "RMSE": rmse_both, "MAE": mae_both, "N": int(len(df))},
        "preds_csv": str(df_path),
    }
    return metrics

def proxy_influence_eval(graphs, scalers, model, out_dir: Path, max_graphs: int = 3):
    rows = []
    for k, (acc, pdbid, chain, G) in enumerate(graphs[:max_graphs]):
        sweep = validate_influence_edges_sweep(model, G, scalers, eps_list=(0.1, 0.25, 0.5, 1.0), sign_tau=1e-5)
        srow = {
            "uniprot": acc, "pdb_id": pdbid, "chain": chain,
            "mean_abs_dA_meas_mean": float(sweep["mean_abs_dA_meas"].mean()),
            "spearman_mean":         float(sweep["spearman"].replace([np.inf, -np.inf], np.nan).dropna().mean())
                                     if "spearman" in sweep.columns else np.nan,
        }
        rows.append(srow)
        sweep.to_csv(out_dir / f"influence_sweep_{acc or 'UNK'}_{k}.csv", index=False)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df_path = out_dir / "influence_proxy_val.csv"
    df.to_csv(df_path, index=False)
    return {
        "proxy_mean_abs_dA_meas": float(df["mean_abs_dA_meas_mean"].mean()),
        "proxy_spearman_mean": float(df["spearman_mean"].mean()),
        "details_csv": str(df_path),
    }

# -----------------------------
# Variant-level (DMS) helpers
# -----------------------------
def load_dms_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"pos","wt","mut","score"}
    missing = req - set(c.lower() for c in df.columns)
    # Try case-insensitive mapping
    colmap = {c.lower(): c for c in df.columns}
    if missing:
        raise ValueError(f"DMS CSV must have columns {req}, got {set(df.columns)}")
    df = df.rename(columns={colmap["pos"]:"pos", colmap["wt"]:"wt", colmap["mut"]:"mut", colmap["score"]:"score"})
    df = df.dropna(subset=["pos","wt","mut","score"]).reset_index(drop=True)
    df["pos"] = df["pos"].astype(int)
    return df

def build_mut_graph(G, pos: int, mut: str, lap_eps: float = 1e-3):
    H = clone_graph(G)
    set_node_aa(H, pos, mut)  # update residue name
    # refresh PRS → g_mem locally
    Lp = laplacian_pinv_weighted(H, eps=lap_eps)
    flux = prs_edge_flux(H, Lp, [pos], [min(pos+5, H.number_of_nodes()-1)])
    adapt_conductance(H, flux, alpha=0.1, key="g_mem", base=1.0)
    return H

def finetune_on_dms_csv(model, G, scalers, csv_path: str, steps: int = 1500, lr: float = 1e-3, clip: float = 1.0,
                        eval_frac: float = 0.2, seed: int = 42, out_dir: Path = None):
    import torch
    rng = np.random.default_rng(seed)
    df = load_dms_csv(csv_path)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_eval = max(1, int(round(eval_frac * len(df))))
    eval_idx = idx[:n_eval]
    train_idx = idx[n_eval:]

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # --- train
    for _ in range(steps):
        b = rng.choice(train_idx, size=min(16, len(train_idx)), replace=False)
        feats, tgt = [], []
        for ii in b:
            pos = int(df.iloc[ii]["pos"]); mut = str(df.iloc[ii]["mut"])
            y   = float(df.iloc[ii]["score"])
            H = build_mut_graph(G, pos, mut)
            X, edges, E = to_features_device(H, scalers, device="cpu")
            feats.append((X,edges,E)); tgt.append([y, 0.0])
        tgt = torch.tensor(tgt, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        yhat = model(feats)
        loss = torch.nn.functional.mse_loss(yhat[:,0:1], tgt[:,0:1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        opt.step()

    # --- eval (variant-level correlations)
    y_true, y_pred = [], []
    for ii in eval_idx:
        pos = int(df.iloc[ii]["pos"]); mut = str(df.iloc[ii]["mut"])
        y   = float(df.iloc[ii]["score"])
        H = build_mut_graph(G, pos, mut)
        X, edges, E = to_features_device(H, scalers, device="cpu")
        with torch.no_grad():
            yh = model([(X,edges,E)])[0]
        y_true.append(y); y_pred.append(float(yh[0].cpu()))
    m = regression_metrics(np.array(y_true), np.array(y_pred))
    if out_dir is not None:
        pd.DataFrame({"pos":df.iloc[eval_idx]["pos"].values,
                      "mut":df.iloc[eval_idx]["mut"].values,
                      "score_true":y_true, "score_pred":y_pred}).to_csv(out_dir/"dms_eval_predictions.csv", index=False)
        with open(out_dir/"dms_eval_metrics.json","w") as f:
            json.dump(m, f, indent=2)
    return m

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", nargs="+", required=True, help="Search terms (UniProt query).")
    ap.add_argument("--n_structures", type=int, default=20, help="Cap structures per run.")
    ap.add_argument("--train_steps", type=int, default=300, help="Training steps.")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction.")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--min_sep", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--lap_eps", type=float, default=1e-3)
    ap.add_argument("--route_w", type=float, default=0.3)
    ap.add_argument("--align_w", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--pdb_cache", type=str, default="./pdb_cache")
    ap.add_argument("--afdb_cache", type=str, default="./afdb_cache")
    ap.add_argument("--save_model", type=str, default="", help="Optional path to save model .pt")

    # Optional DMS fine-tune
    ap.add_argument("--dms_csv", type=str, default="", help="CSV with columns: pos, wt, mut, score (single protein).")
    ap.add_argument("--dms_steps", type=int, default=1500)
    ap.add_argument("--dms_lr", type=float, default=1e-3)
    ap.add_argument("--dms_eval_frac", type=float, default=0.2)

    # Optional ablation + RL on first val graph
    ap.add_argument("--ablate_k", type=int, default=10)
    ap.add_argument("--ablate_eps", type=float, default=0.5)
    ap.add_argument("--rl_steps", type=int, default=0, help=">0 to run RL refine g_mem on first val graph")
    ap.add_argument("--rl_lr", type=float, default=0.5)
    ap.add_argument("--rl_lambda_ddg", type=float, default=0.5)

    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect proteins
    accs = collect_uniprot_via_terms(args.terms, size=5)
    if len(accs) == 0:
        raise RuntimeError("No UniProt accessions collected from the provided terms.")
    print("Collected UniProt:", len(accs), accs)

    # 2) Map to structures
    sel = uniprot_to_structures(accs, cap=args.n_structures)
    sel_path = out_dir / "selected_structures.csv"
    sel.to_csv(sel_path, index=False)
    print("Saved:", sel_path)

    # 3) Build graphs
    graphs = build_graphs_from_structures(
        sel, pdb_cache=args.pdb_cache, afdb_cache=args.afdb_cache,
        topk=args.topk, min_sep=args.min_sep, sigma=args.sigma, lap_eps=args.lap_eps
    )
    if len(graphs) < 2:
        raise RuntimeError(f"Need at least 2 graphs to do an 80/20 split, got {len(graphs)}.")
    print("Graphs built:", len(graphs))

    # 4) Split
    tr_idx, va_idx = split_indices(len(graphs), val_frac=args.val_frac, seed=args.seed)
    train_graphs = [graphs[i] for i in tr_idx]
    val_graphs   = [graphs[i] for i in va_idx]
    split_info = {
        "n_total": len(graphs),
        "train_idx": tr_idx, "val_idx": va_idx,
        "n_train": len(train_graphs), "n_val": len(val_graphs),
    }
    with open(out_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print("Split:", split_info)

    # 5) Fit scalers on TRAIN ONLY (avoid leakage)
    scalers = fit_feature_scalers(train_graphs)
    np.savez(out_dir / "scalers.npz",
             node_mean=scalers["node_mean"], node_std=scalers["node_std"],
             edge_mean=scalers["edge_mean"], edge_std=scalers["edge_std"])

    # 6) Train (route + alignment supervision)
    model = train_graph_regressor_amp_route(
        train_graphs, scalers,
        steps=args.train_steps, route_w=args.route_w, align_w=args.align_w
    )

    # Optionally save weights
    if args.save_model:
        try:
            import torch
            torch.save(model.state_dict(), args.save_model)
            print("Saved model:", args.save_model)
        except Exception as e:
            print("WARN: could not save model:", e)

    # 7) Evaluate graph-level (train + val)
    metrics = {}
    metrics["train"] = evaluate_split("train", train_graphs, scalers, model, out_dir)
    metrics["val"]   = evaluate_split("val",   val_graphs,   scalers, model, out_dir)

    # 8) Proxy influence diagnostic on val (quick)
    metrics["val_proxy_influence"] = proxy_influence_eval(val_graphs, scalers, model, out_dir, max_graphs=min(3, len(val_graphs)))

    # 9) Optional: edge ablation sanity & RL refine on first val graph
    if len(val_graphs) > 0:
        acc, pdbid, chain, Gv = val_graphs[0]
        # edge ablation before RL
        ab_before = ablate_edges_by_rank(model, Gv, scalers, k=args.ablate_k, eps=args.ablate_eps)
        metrics["val_ablation_before"] = ab_before
        # RL refine (updates g_mem only)
        if args.rl_steps > 0:
            Gv_rl = clone_graph(Gv)
            Gv_rl = rl_refine_gmem(model, Gv_rl, scalers, steps=args.rl_steps, lr=args.rl_lr, lam_ddg=args.rl_lambda_ddg)
            ab_after = ablate_edges_by_rank(model, Gv_rl, scalers, k=args.ablate_k, eps=args.ablate_eps)
            metrics["val_ablation_after"] = ab_after

    # 10) Optional: per-variant fine-tune on a single DMS CSV
    if args.dms_csv:
        # choose the first train graph by default for fine-tune (same protein as CSV)
        G_ft = train_graphs[0][3]
        m_dms = finetune_on_dms_csv(
            model, G_ft, scalers,
            csv_path=args.dms_csv,
            steps=args.dms_steps, lr=args.dms_lr,
            eval_frac=args.dms_eval_frac, seed=args.seed,
            out_dir=out_dir
        )
        metrics["dms_eval"] = m_dms

    # 11) Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 12) Console summary
    def pretty(name, m):
        A, D, C = m["A"], m["ddG"], m["combined"]
        def nz(v): 
            return v if (v==v) else 'nan'
        print(f"\n== {name.upper()} ==")
        print(f"Â:    MSE={A['MSE']:.6f} RMSE={A['RMSE']:.6f} MAE={A['MAE']:.6f} "
              f"R2={nz(A['R2'])} Pearson={nz(A['Pearson'])} Spearman={nz(A['Spearman'])} N={A['N']}")
        print(f"ΔΔĜ:  MSE={D['MSE']:.6f} RMSE={D['RMSE']:.6f} MAE={D['MAE']:.6f} "
              f"R2={nz(D['R2'])} Pearson={nz(D['Pearson'])} Spearman={nz(D['Spearman'])} N={D['N']}")
        print(f"Both:  MSE={C['MSE']:.6f} RMSE={C['RMSE']:.6f} MAE={C['MAE']:.6f} N={C['N']}")

    pretty("train", metrics["train"])
    pretty("val",   metrics["val"])

    if metrics.get("val_proxy_influence"):
        p = metrics["val_proxy_influence"]
        print("\n== VAL (proxy influence) ==")
        print(f"mean_abs_dA_meas (avg over graphs) ~ {p.get('proxy_mean_abs_dA_meas', np.nan)}")
        print(f"spearman (avg over graphs)          ~ {p.get('proxy_spearman_mean', np.nan)}")
        print(f"details CSV: {p.get('details_csv','')}")

    if "val_ablation_before" in metrics:
        print("\n== Edge ablation ΔA (first val graph) ==")
        print("before:", metrics["val_ablation_before"])
        if "val_ablation_after" in metrics:
            print("after :", metrics["val_ablation_after"])

    if "dms_eval" in metrics:
        m = metrics["dms_eval"]
        print("\n== DMS eval (variant-level, held-out) ==")
        print(f"MSE={m['MSE']:.6f} RMSE={m['RMSE']:.6f} MAE={m['MAE']:.6f} "
              f"R2={m['R2'] if m['R2']==m['R2'] else 'nan'} "
              f"Pearson={m['Pearson'] if m['Pearson']==m['Pearson'] else 'nan'} "
              f"Spearman={m['Spearman'] if m['Spearman']==m['Spearman'] else 'nan'} "
              f"N={m['N']}")


if __name__ == "__main__":
    main()

