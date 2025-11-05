# Res-Contact — ESM2-based Protein Contact Prediction (Laptop-friendly)

> **One-line**: Freeze ESM2 to get per-residue embeddings, train a tiny head that predicts Cα–Cα ≤ 8 Å contacts from a single sequence. Optional light MSA features and **optional** homology-derived structural priors (templates) can be fused in. PSI drift monitoring is **batch-only** for now.

---

## 0) Highlights & scope

- **Backbone**: `facebook/esm2_t6_8M_UR50D` (frozen; no fine-tuning) → per-residue H ∈ ℝ^{L×320}
- **Head**: Linear → ReLU → Dropout → Bilinear + learnable distance-bias (|i−j| bins)
- **Labels**: PDB/mmCIF Cα distances @ **8 Å** (strict upper triangle; diagonal masked)
- **MSA**: **Optional** lightweight 1‑D (+21 dims: AA freqs + entropy); zeros when missing
- **Homology templates**: **Optional** (MMseqs2 + PDB/AFDB) → structural **priors** fused with ESM2 head (no backbone fine‑tuning)
- **Monitoring**: PSI **batch-only** (no live endpoints yet)
- **Device**: Apple MPS preferred, CPU fallback; designed for **8‑GB MacBook Air (M3)**
- **Deployment**: FastAPI present; **containerization & cloud deploy are future work**

This README is consistent with **Report.docx**, **roadmap.txt**, and **Roadmap.xlsx**:
- ESM embeddings with **8 Å** contact definition
- MSA is **optional** (laptop constraint)
- PSI monitoring is **batch** now, **streaming** planned
- FastAPI exists; Docker/Cloud deploy listed under **Future work** only

---

## 1) Install

```bash
# Recommended: a fresh conda/venv
conda create -n rescontact python=3.11 -y
conda activate rescontact

pip install -r requirements.txt
# If you use Optuna sweeps:
pip install optuna sqlalchemy

# Apple MPS friendliness (optional)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

**Note (macOS / MPS):**
- If you hit `NotImplementedError: aten::triu_indices on MPS`, enable CPU fallback for that op:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- If `torchvision` warns about `libjpeg`, you may ignore unless you use `torchvision.io`.

**Note (typing_extensions vs tensorflow-macos):**
- If you upgrade `typing_extensions` for SQLAlchemy/Optuna and see conflicts with `tensorflow‑macos`, keep the sweep in a **separate env** to avoid pinning issues.

---

## 2) Repository layout (key parts)

```
Res-contact/
├─ configs/
│  ├─ rescontact.yaml                 # default config (ESM-only w/ optional MSA)
│  └─ rescontact.tuned.yaml           # (optional) tuned config
├─ sweeps/
│  ├─ rescontact.db                   # Optuna sqlite study (after running)
│  └─ *.tuned.yaml                    # saved best configs (after sweeps)
├─ scripts/
│  ├─ train.py                        # train (full-grid head) — uses BCEWithLogits
│  ├─ eval.py                         # eval & metrics (PR/ROC/F1; optional P@L)
│  ├─ build_baseline.py               # PSI baseline (quantile bins on train)
│  ├─ monitor_eval.py                 # compute PSI & histos per split (batch)
│  ├─ retrieve_homologs.py            # (optional) MMseqs2 retrieval
│  ├─ build_template_priors.py        # (optional) PDB/AFDB priors from homologs
│  └─ fuse_priors.py                  # (optional) visualize/inspect priors
├─ src/rescontact/
│  ├─ api/server.py                   # FastAPI app (/predict, /visualize) — batch PSI future
│  ├─ data/loader.py                  # PDB/mmCIF parsing; masks & labels @ 8Å
│  ├─ features/embedding.py           # ESM2 embedding cache (frozen backbone)
│  ├─ model/head.py                   # Linear→ReLU→Dropout→Bilinear + dist-bias
│  ├─ templates/                      # (optional) homology templates subsystem
│  │  ├─ mmseqs.py                    # wrapper for MMseqs2
│  │  ├─ mapping.py                   # hit→query residue index mapping
│  │  ├─ features.py                  # build distance/contact prior channels
│  │  ├─ fuse.py                      # fusion (logit_blend / feature_concat)
│  │  └─ template_db.py               # helpers to find/open PDB/AFDB files
│  └─ utils/metrics.py                # PR-AUC, ROC-AUC, F1, (optional P@L)
├─ .cache/rescontact/                 # embedding & feature caches
├─ reports/                           # PSI json/png artifacts (batch)
└─ README.md
```

---

## 3) Configuration (ESM-only default)

```yaml
# configs/rescontact.yaml (snippet)
model:
  esm_model: facebook/esm2_t6_8M_UR50D
  embed_dim: 320
  head_hidden: 256
  dropout_p: 0.1
  dist_bias_bins: 512

features:
  use_msa: false          # if true and available, +21 dims (freqs+entropy)
  msa_max_seqs: 256       # caps if you enable MSA retrieval
  cache_dir: .cache/rescontact

data:
  pdb_root: data/pdb      # train/val/test structure roots
  max_len_per_chain: 600  # crop long chains to fit RAM
  include_inter_chain: true

train:
  batch_size: 1
  lr: 1.5e-3
  weight_decay: 1e-4
  epochs: 20
  early_stop_patience: 5
  pos_weight: 10.0        # optional positive up-weighting

eval:
  threshold: 0.50         # decision threshold for F1 (PR/ROC are threshold-free)
  report_pal: false       # P@L is optional (prefer long-range P@k in future)

monitor:
  psi:
    enabled: true         # batch-only
    bins: 20
    baseline_path: monitor/baseline.json
```

**Environment knobs** you may set:
```bash
export MAX_LEN_PER_CHAIN=600
export RESCONTACT_VERBOSE=1
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1   # only if you hit unsupported MPS ops
```

---

## 4) Quickstart

### 4.1 Train (ESM-only; MSA optional)
```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/rescontact.yaml
```

### 4.2 Evaluate
```bash
PYTHONPATH=src python scripts/eval.py \
  --config configs/rescontact.yaml \
  --ckpt checkpoints/model_best.pt \
  --split test \
  --max_test_examples 500
```

**Metrics reported**: PR‑AUC, ROC‑AUC, F1 on masked upper triangle (i<j).  
**Note on P@L**: Computed over all separations it can be inflated by near-diagonal pairs; prefer long‑range P@k in future.

---

## 5) Monitoring (PSI drift) — batch only

Build the **baseline** (quantile bins + proportions **from train**):
```bash
PYTHONPATH=src python scripts/build_baseline.py \
  --config configs/rescontact.yaml \
  --out monitor/baseline.json \
  --max_examples 200
```

After each eval, write PSI & plots under `reports/`:
```bash
PYTHONPATH=src python scripts/monitor_eval.py \
  --config configs/rescontact.yaml \
  --split val \
  --baseline monitor/baseline.json
```

Artifacts per run:
- `reports/psi_<split>_<ts>.json` – PSI value, category, live proportions, metadata
- `reports/score_<split>_<ts>.png` – probability histogram
- `reports/length_<split>_<ts>.png` – sequence length distribution
- `reports/sep_<split>_<ts>.png` – |i−j| separation distribution

**Thresholds** (configurable): PSI ≤ 0.10 **stable**, 0.10–0.25 **watch**, > 0.25 **drift**.  
**Streaming PSI**: *future work* (not in this server version).

---

## 6) Hyperparameter tuning (Optuna) — optional

```bash
export MAX_LEN_PER_CHAIN=900
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python optuna_sweep.py \
  --config configs/rescontact.yaml \
  --script scripts/train.py \
  --study sqlite:///sweeps/rescontact.db --study-name esm8m_local \
  --trials 12 --epochs 24 --batch-size 1 \
  --min-train-examples 1000 --val-split 0.8 \
  --tune-hidden   --space-hidden 128 160 192 256 \
  --tune-lr       --space-lr 0.0010 0.0012 0.0015 0.0018 \
  --tune-dropout  --space-dropout 0.0 0.1 0.2 \
  --tune-threshold --thresh-min 0.28 --thresh-max 0.38 --thresh-step 0.01 \
  --objective bf1 --pruner none \
  --logs-dir sweeps/logs \
  --save-best-config sweeps/rescontact.tuned.yaml
```

> If SQLAlchemy requires newer `typing_extensions`, consider running sweeps in a **separate env** to avoid `tensorflow‑macos` pin conflicts.

---

## 7) Optional: Homology‑augmented structural priors (templates)

**Goal**: Re‑use ESM2 embeddings **and** pull structural priors from **homologous** proteins (PDB/AFDB) found via MMseqs2. This is **not** fine‑tuning and **not** RAG; the backbone stays frozen, and we fuse priors with the head.

### 7.1 Requirements
- MMseqs2 installed (`mmseqs` on PATH)
- Local PDB mirror or RCSB fetch; AFDB `.cif`/`.pdb` via download
- Minimal disk: template cache under `data/templates/cache/`

### 7.2 Enable in config
```yaml
templates:
  enabled: true
  mmseqs:
    mode: easy-search          # or prefetched index
    db: data/templates/mmseqs/uniref90
    max_hits: 8
    min_ident: 0.3
    min_cov: 0.6
  sources:
    pdb: true
    afdb: true
  features:
    use_distogram: true        # discretized dist bins to prior channels
    use_contact: true          # binary/soft contact prior
    use_confidence: true       # pLDDT/PAE weights if AFDB available
  fusion:
    method: logit_blend        # 'logit_blend' (simple) or 'feature_concat'
    alpha: 0.35                # blend weight for template logits
  cache_dir: data/templates/cache
```

### 7.3 Pipeline
1) **Retrieve homologs**  
   ```bash
   PYTHONPATH=src python scripts/retrieve_homologs.py \
     --fasta data/queries/query.fasta \
     --out data/templates/mmseqs_hits.json \
     --db data/templates/mmseqs/uniref90
   ```
2) **Build priors** (download/open PDB/AFDB, map residues, make NPZ priors)  
   ```bash
   PYTHONPATH=src python scripts/build_template_priors.py \
     --hits data/templates/mmseqs_hits.json \
     --out data/templates/cache
   ```
3) **Train with fusion enabled** (head consumes ESM2 + priors)  
   ```bash
   PYTHONPATH=src python scripts/train.py \
     --config configs/rescontact.yaml
   ```
4) **Eval** as usual; PSI/metrics pipelines unchanged.

> On a laptop: keep `max_hits` small (e.g., 4–8), cache aggressively, and prefer `logit_blend` first.

---

## 8) Serving (local)

Start the FastAPI app:
```bash
PYTHONPATH=src uvicorn src.rescontact.api.server:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /predict` — sequence or FASTA path → probabilities (optionally binary via threshold)
- `GET  /visualize` — returns a PNG heatmap for quick inspection
- `GET  /healthz` — lightweight health check

> PSI **streaming endpoints** are **future work**; current PSI is **batch-only** via scripts above.  
> **Containerization & cloud deploy (GCP/AWS)** are **future work** and intentionally omitted from roadmap files.

---

## 9) FAQ

- **Is this fine‑tuning?** No. ESM2 is **frozen** and used as a feature extractor.
- **Is this RAG?** No. PDB/AFDB priors are precomputed features, not retrieval‑augmented generation.
- **Why cache embeddings?** ESM2 forward is the most expensive step; cache by `(model_id, seq_hash, crop)`.
- **Why is P@L near 1.0 sometimes?** Without long‑range filtering, near‑diagonal pairs dominate. Prefer long‑range P@k later.

---

## 10) Troubleshooting (quick)

- `aten::triu_indices` not on MPS → `export PYTORCH_ENABLE_MPS_FALLBACK=1`
- `typing_extensions` conflicts → run Optuna/sweeps in a separate env
- Long chains OOM → lower `max_len_per_chain`, or trim long chains
- Embedding time too long → warm the cache first, then train/eval

---

## 11) License
MIT (or your org’s standard).

---

## 12) Citations
- Rives et al., *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences*. (ESM)
- Jumper et al., *Highly accurate protein structure prediction with AlphaFold*. (AFDB references)
