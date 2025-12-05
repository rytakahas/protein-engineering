# ResIntNet — Residue‑Interaction Network with **Adaptive Memory** (g_mem)

A compact, Mac‑friendly package that builds residue–residue graphs from PDB/AFDB structures, injects a **memory** term on edges (g_mem) via PRS‑style flux, and trains a lightweight graph regressor so **model outputs depend on g_mem**. Includes:
- Route supervision (matches a g_mem gate to PRS flux)
- Gradient‑alignment loss (aligns ∂Â/∂g_mem with PRS flux)
- Optional **RL refinement** to push g_mem toward higher predicted activity with stability penalty
- Diagnostics: edge **ablation** ΔA, **influence sweep**, toy gradient check

Runs on an **8 GB MacBook Air (M3)** (CPU/MPS). Also works on Linux/Colab/Kaggle CPU/GPU.

---

## What’s new in the latest notebook→package refactor

**Plumbing fixes so g_mem actually matters:**
- Keep **raw** `g_mem` (don’t z‑score column 2 of edge features).
- **Linear/softplus gate**: `gate = clamp(1 + gate_scale * g_mem, 0, 3)` with learnable `gate_scale`.
- Edge **context skip** into the readout so outputs depend directly on edge features weighted by g_mem.
- Two supervision signals:
  - **route loss**: normalized gate ≈ PRS edge‑flux
  - **align loss**: normalized saliency `|∂Â/∂g_mem|` ≈ PRS edge‑flux
- Optional **RL loop** that updates g_mem by ascending the reward `Â − λ · ReLU(ΔΔĜ)`.

**You should now see (as in your logs):**
- Toy check: mean `|∂Â/∂g_mem| ≥ 1e‑3`
- Non‑zero ΔA after edge perturbations/ablations
- Influence sweep with positive Spearman for reasonable ε (e.g., 0.25–1.0)

---

## Install

```bash
# Python 3.10–3.12 recommended
python -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel

# Inside the repo root (where pyproject.toml lives)
pip install -e .

# Optional (nicer progress bars in notebooks):
pip install ipywidgets
```

**Dependencies used by the package:** `torch`, `numpy`, `pandas`, `scipy`, `networkx`, `biopython`, `requests`, `tqdm` (installed via `pip install -e .`).

---

## Directory layout

```
resintnet/
├─ scripts/
│  ├─ train_memory.py          # CLI: train + eval (+ optional RL) with 80/20 split
│  ├─ rank_mutations.py        # CLI: rank candidate mutations after training
│  └─ ingest_mutations.py      # helper to format DMS CSV (pos, wt, mut, score)
├─ src/resintnet/
│  ├─ __init__.py
│  ├─ graph.py                 # parsing, graph build, features, normalization
│  ├─ memory_flow.py           # data collection, training loop, diagnostics, RL
│  ├─ prs.py                   # PRS helpers (used inside memory_flow.py)
│  ├─ models/sage.py           # GraphBatchNetAMP with g_mem gate + edge context
│  └─ ingest/
│     ├─ __init__.py           # wire your external labels here if you have them
│     └─ adapters/
│        ├─ d3distal.py        # (optional) loader for custom CSVs
│        └─ generic_csv.py     # (optional) generic CSV loader
└─ notebook/                   # demo notebooks
```

---

## Quick start (small run)

```bash
python scripts/train_memory.py   --terms TP53 LYSOZYME   --n_structures 20   --train_steps 200   --out ./outputs_small
```

**Outputs (in `--out`):**
- `selected_structures.csv` – UniProt → (best PDB chain or AFDB)
- `scalers.npz` – feature means/stds (fit on **train only**)
- `split.json` – indices for 80/20 split
- `predictions_train.csv`, `predictions_val.csv`
- `metrics.json` – MSE/RMSE/MAE/R²/Pearson/Spearman per head + combined
- `influence_sweep_<acc>_k.csv` (a few per‑graph CSVs, validation only)
- `influence_proxy_val.csv` – summary over validation set

**Interpreting metrics:**
- Correlations (Pearson/Spearman) can be `nan` with **weak or degenerate labels** (e.g., if your loaders return constant means). Focus on proxy influence and ablations, or add **per‑variant DMS supervision** later.

---

## Stronger run (with RL refinement)

```bash
python scripts/train_memory.py   --terms TP53 LYSOZYME   --n_structures 30 --train_steps 400   --route_w 0.5 --align_w 0.5   --rl_steps 150 --rl_lr 1.2 --rl_lambda_ddg 0.3   --out ./outputs_stronger_rl
```

You should see logs like:

```
[RL t=80] mean|grad|=5.53e-03  sat_lo=0.00  sat_hi=0.30  ΔA_top=0.0578  ΔA_bot=0.0016
[RL t=149] mean|grad|=7.15e-03  sat_lo=0.00  sat_hi=0.30  ΔA_top=0.0879  ΔA_bot=0.0553
```

…and **edge ablation gains** on the validation graph, e.g.

```
before: {'top': 0.0177, 'bottom': 0.0041}
after : {'top': 0.0655, 'bottom': 0.0124}
```

---

## Broader run (more proteins)

```bash
python scripts/train_memory.py   --terms TP53 LYSOZYME DHFR HSP90 UBIQUITIN   --n_structures 40 --val_frac 0.33   --train_steps 600 --route_w 0.5 --align_w 0.5   --out ./outputs_broader
```

If your network bandwidth or RAM is limited on a Mac, **lower `--n_structures`** and/or `--train_steps`. Everything scales approximately linearly.

---

## CLI reference (most useful flags)

```
--terms <STR ...>           # UniProt search terms (e.g., TP53, LYSOZYME)
--n_structures INT          # max structures to consider
--val_frac FLOAT            # validation fraction (default 0.2)
--train_steps INT           # training iterations
--route_w FLOAT             # weight of route loss
--align_w FLOAT             # weight of gradient‑alignment loss
--topk INT --min_sep INT    # graph construction knobs
--sigma FLOAT --lap_eps FLOAT

# caches and outputs
--pdb_cache PATH
--afdb_cache PATH
--out PATH
--save_model PATH.pt        # optional: save weights

# RL refinement (optional)
--rl_steps INT              # iterations of RL after training
--rl_lr FLOAT               # step size for g_mem updates
--rl_lambda_ddg FLOAT       # penalty on predicted ΔΔG >= 0

# (Some versions expose ablation flags)
--ablate_k INT --ablate_eps FLOAT
```

---

## Programmatic usage (Python)

```python
from resintnet.memory_flow import (
    collect_uniprot_via_terms, uniprot_to_structures, build_graphs_from_structures,
    fit_feature_scalers, train_graph_regressor_amp_route, predict_graph_amp,
    validate_influence_edges_sweep, ablate_edges_by_rank, rl_refine_gmem
)

accs = collect_uniprot_via_terms(["TP53","LYSOZYME"], size=5)
sel  = uniprot_to_structures(accs, cap=20)
graphs = build_graphs_from_structures(sel, pdb_cache="./pdb_cache", afdb_cache="./afdb_cache")
scalers = fit_feature_scalers(graphs[: int(0.8*len(graphs))])
model = train_graph_regressor_amp_route(graphs[: int(0.8*len(graphs))], scalers,
                                        steps=300, route_w=0.5, align_w=0.5)

# Diagnostics on a graph
uni, pdbid, chain, G = graphs[-1]
sweep = validate_influence_edges_sweep(model, G, scalers)
print(sweep)

print( ablate_edges_by_rank(model, G, scalers, k=10, eps=0.5) )

# Optional RL refinement of g_mem in‑place
G_rl = rl_refine_gmem(model, G, scalers, steps=50, lr=0.5, lam_ddg=0.5)
```

---

## Sanity checks (what “working” looks like)

- **Toy gradient** (internal check in notebooks): mean `|∂Â/∂g_mem|` ≳ **1e‑3**.
- **Influence sweep** (validation): non‑zero `mean_abs_dA_meas`; positive Spearman for mid‑range ε (0.25–1.0).
- **Edge ablation**: changing top‑ranked edges by +ε should move Â more than bottom edges; after RL, the gap should widen.

If any of these are **exactly 0**, see troubleshooting.

---

## Troubleshooting

**All zeros in sweeps / ∂Â/∂g_mem ~ 0**
- Ensure `to_features_device(..., requires_grad=True)` is used where gradients are read.
- Confirm we **do not** z‑score `g_mem` (edge column 2). The package keeps column 2 raw.
- Increase gate sensitivity: try `route_w=0.5 align_w=0.5` and in code, `gate_scale` initial value ≥ **2.0**.
- Use a **linear/softplus** gate, not sigmoid in the flat region (already implemented).

**`RuntimeError: One of the differentiated Tensors does not require grad`**
- Make sure the E tensor is created with `requires_grad=True` before forward.
- Avoid `detach()` on features; keep the forward call enclosed until `.backward()`.

**`Error displaying widget: model not found` in notebooks**
- Purely cosmetic (ipywidgets missing). `pip install ipywidgets` to fix the display.

**PDBe coverage/resolution `None` → type errors**
- The selector is robust: non‑finite values default to safe fallbacks (e.g., resolution 9.9 Å).

**Memory on 8 GB Mac**
- Lower `--n_structures`, `--train_steps`, use `topk=6..8`, `min_sep=2..3`.
- Prefer AFDB when PDBe download is slow/unavailable (handled automatically).

---

## External labels (optional, later)

Hook real labels by editing **`src/resintnet/ingest/__init__.py`**:

```python
def get_mavedb_first_scores_for(acc):  # return DataFrame with 'score' or None
    return None  # wire your loader here

def get_fireprot_labels_for(acc):      # return DataFrame with 'ddG' col or None
    return None  # wire your loader here
```

Even **one** DMS table gives thousands of per‑variant targets and will improve generalization.

---

## Reproducibility

- We set NumPy/Python seeds in the scripts. For strict reproducibility across PyTorch/CPU/MPS, also set:
  - `torch.use_deterministic_algorithms(True)` and relevant env vars if needed (off by default due to speed).

---

## Citations & background (informal)
- PRS and effective resistance ideas for signal routing over graphs
- Protein language models and equivariant GNNs (future integration points)

---

## License
Apache‑2.0 (or your choice).

---

## Quick recipes you already ran (from your logs)

Small set:
```bash
python scripts/train_memory.py   --terms TP53 LYSOZYME   --n_structures 20 --train_steps 200   --out ./outputs_small
```

Stronger with RL:
```bash
python scripts/train_memory.py   --terms TP53 LYSOZYME   --n_structures 30 --train_steps 400   --route_w 0.5 --align_w 0.5   --rl_steps 150 --rl_lr 1.2 --rl_lambda_ddg 0.3   --out ./outputs_stronger3_rl
```
