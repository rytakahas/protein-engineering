## Res-Contact — ESM2-based Residue–Residue Contact Prediction (Laptop-friendly)

This repo trains a **lightweight contact predictor** on top of a **frozen ESM2** encoder and produces
probability (and optionally binary) contact maps from a **single input sequence**. It supports **optional 1‑D MSA features** and, when enabled, **homology template priors** fetched via a **server‑only MMseqs2 path** (no local UniRef DB required).

> **Design choices (kept consistent with Report & Roadmap)**
> - **ESM2 backbone is frozen** (feature extractor), not fine‑tuned.
> - **Cα–Cα ≤ 8.0 Å** defines a contact. Upper triangle, mask-aware loss/metrics.
> - **MSA is optional** (off by default on laptops); when unavailable the +21 dims are zeros.
> - **PSI monitoring is batch-only** in this version; live/streaming PSI is noted as future work.
> - Dockerization is a **future** item; not part of the 7‑day roadmap files.

---

### 1) Setup

### Requirements
- Python 3.10–3.11
- macOS with Apple Silicon **MPS** support recommended (8‑GB MacBook Air M3 OK for demo scale)
- PyTorch, Biopython, scikit‑learn, Optuna, etc. (see `requirements.txt`)

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: for Apple MPS stability
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Environment (optional)
```bash
export RESCONTACT_CACHE_DIR=".cache/rescontact"
export RESCONTACT_TEMPLATE_DIR="$RESCONTACT_CACHE_DIR/templates"
mkdir -p "$RESCONTACT_CACHE_DIR" "$RESCONTACT_TEMPLATE_DIR"
```

---

### 2) Data layout

```
data/
├─ fasta/
│  └─ demo.fasta              # one or multiple query sequences
├─ pdb/
│  ├─ train/                  # PDB/mmCIF structures used for training labels
│  └─ test/                   # PDB/mmCIF structures used for testing labels
└─ msa/                       # optional .a3m files (if you enable MSA)
```

- **Ground truth** is built from PDB/mmCIF: per chain we extract sequence + Cα coords, mask missing residues, and label **Y[i,j]=1** if **dist_Cα(i,j) ≤ 8.0 Å** (strict upper triangle).

---

### 3) Train / Eval (ESM‑only, default)

**Config:** `configs/rescontact.yaml` (ESM‑only by default; MSA off).

```bash
# Train
PYTHONPATH=src python scripts/train.py \
  --config configs/rescontact.yaml \
  --save-dir artifacts/esm8m_baseline \
  --epochs 20 --batch-size 1

# Evaluate
PYTHONPATH=src python scripts/eval.py \
  --config configs/rescontact.yaml \
  --ckpt artifacts/esm8m_baseline/model_best.pt \
  --split test
```

**Notes**
- The head uses **BCEWithLogitsLoss** on the masked **upper triangle**.
- Metrics: **PR‑AUC, ROC‑AUC, F1** (optional P@L).
- Probability map: `P = sigmoid(logits)`; diagonal zeroed; mask applied; optional symmetrization for display.

---

### 4) Optional MSA (1‑D features)

If you switch `features.use_msa: true`, the dataloader appends **+21 dims** per residue (20 AA frequencies + entropy). If an MSA is **not** present for a sequence, those dims are **zeros** (shape‑stable).

**Check MSA coverage quickly**
```bash
PYTHONPATH=src python scripts/check_msa.py --fasta data/fasta/demo.fasta
```

MSA remains **optional** to keep the laptop workflow light.

---

### 5) Monitoring (PSI drift) — batch only (current)

Compute **Population Stability Index (PSI)** on predicted probabilities over the masked upper triangle (i<j).

**Build a baseline (train quantile bins)**
```bash
PYTHONPATH=src python scripts/build_baseline.py \
  --config configs/rescontact.yaml \
  --out monitor/baseline.json \
  --max_examples 200
```

**Run PSI on an evaluation split**
```bash
PYTHONPATH=src python scripts/monitor_eval.py \
  --config configs/rescontact.yaml \
  --ckpt artifacts/esm8m_baseline/model_best.pt \
  --split test \
  --baseline monitor/baseline.json \
  --out-dir reports/
```

Artifacts under `reports/` include: `psi_<split>_<ts>.json`, `score_<split>_<ts>.png`, `length_<split>_<ts>.png`, `sep_<split>_<ts>.png`.  
Thresholds (configurable): **≤0.10 stable**, **0.10–0.25 watch**, **>0.25 drift**.  
Live/streaming PSI via API endpoints is **future work**.

---

### 6) Hyperparameter tuning (Optuna, laptop‑friendly)

A coarse sweep is supported, but keep trials/epochs small on 8‑GB RAM.

```bash
export MAX_LEN_PER_CHAIN=900
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python optuna_sweep.py \
  --config configs/rescontact.yaml \
  --script scripts/train.py \
  --study sqlite:///sweeps/rescontact.db --study-name esm8m_local \
  --trials 12 --epochs 24 --batch-size 1 \
  --tune-hidden   --space-hidden 128 160 192 256 \
  --tune-lr       --space-lr 0.0010 0.0012 0.0015 0.0018 \
  --tune-dropout  --space-dropout 0.0 0.1 0.2 \
  --tune-threshold --thresh-min 0.28 --thresh-max 0.38 --thresh-step 0.01 \
  --objective bf1 --pruner none \
  --logs-dir sweeps/logs \
  --save-best-config sweeps/rescontact.tuned.yaml
```

> If you hit `typing_extensions` import errors from SQLAlchemy/Optuna, upgrade:
> `pip install -U "typing_extensions>=4.12.0"` (note: TensorFlow‑macOS may pin `<4.6.0`).

---

### 7) Homology templates — server‑only MMseqs2 (no local DB)

This path **does not require local UniRef/UniProt**. Everything is **download‑once and cache‑first**.  
To **train with templates**, you **must build template priors _before_ training**.

### Env
```bash
export MMSEQS_SERVER_URL="https://your-mmseqs-server"   # remote MMseqs2 endpoint
export RESCONTACT_CACHE_DIR=".cache/rescontact"
export RESCONTACT_TEMPLATE_DIR="$RESCONTACT_CACHE_DIR/templates"
mkdir -p "$RESCONTACT_TEMPLATE_DIR/priors"
```

#### 1) Retrieve homologs (cached JSON hits)
Writes a compact JSON with top hits (identity/coverage) for each query in your FASTA.
```bash
PYTHONPATH=src python scripts/retrieve_homologs.py \
  --fasta data/fasta/demo.fasta \
  --out data/templates/mmseqs_hits.json \
  --server-url "$MMSEQS_SERVER_URL" \
  --db uniref90 --max-hits 8 --min-ident 0.30 --min-cov 0.60
```

#### 2) Build template priors (cached structures)
Downloads only the structures needed for those hits (from **PDB**/**AFDB**), maps them to the query indices,
and writes **per-query prior channels** (contact/distance bins) into the cache.
```bash
PYTHONPATH=src python scripts/build_template_priors.py \
  --hits data/templates/mmseqs_hits.json \
  --query-fasta data/fasta/demo.fasta \
  --out-dir "$RESCONTACT_TEMPLATE_DIR/priors" \
  --structure-source "pdb,afdb" \
  --max-hits-per-query 8 \
  --max-downloads-per-run 50
```

**Notes**
- Everything is keyed by **(query sequence hash, hit accession, model)** and re‑used on subsequent runs.
- If no homolog structures are found, training/eval **fallback to ESM‑only** seamlessly.

#### 3) Enable templates in config and train
`configs/rescontact.yaml` (snippet):
```yaml
features:
  use_templates: true              # enable template priors
  use_msa: false                   # optional; keep off on laptop

templates:
  provider: "mmseqs_remote"        # server-only path
  server_url: "${MMSEQS_SERVER_URL}"
  db: "uniref90"
  min_ident: 0.30
  min_cov: 0.60
  max_hits: 8
  cache_dir: "${RESCONTACT_TEMPLATE_DIR}"

  # Fusion with ESM head:
  fuse_mode: "logit_blend"         # ["logit_blend", "feature_concat"]
  blend_alpha: 0.3                 # 0.0=ignore templates; 1.0=templates dominate
```

**Train**
```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/rescontact.yaml \
  --save-dir artifacts/esm8m_templates \
  --epochs 20 --batch-size 1
```

**Eval**
```bash
PYTHONPATH=src python scripts/eval.py \
  --config configs/rescontact.yaml \
  --ckpt artifacts/esm8m_templates/model_best.pt \
  --split test
```

**How this meets the task**  
MMseqs2 finds **similar sequences**; PDB/AFDB provides **structural contacts** for those hits; we map residues to the query and build **template priors** that the contact head **fuses** with ESM predictions (either by **logit blending** or **feature concatenation**). This incorporates **structural data from similar sequences** into training and inference.

**Pitfalls**
- **Order matters**: `retrieve_homologs → build_template_priors → train/eval`.
- FASTA sequences must match the sequences used for label construction after any cropping.
- Keep `max-hits` small on laptops; everything is cached for reuse.

---

### 8) Caching & storage

- **Embeddings**: `src/rescontact/features/embedding.py` saves per‑sequence `H ∈ ℝ^{L×d}` as `.npz` under `${RESCONTACT_CACHE_DIR}` with keys `(model_id, seq_hash, crop)`.
- **MSA features**: saved alongside embeddings; when missing, the 21 dims are zeros.
- **Template priors**: under `${RESCONTACT_TEMPLATE_DIR}/priors`, keyed by query hash and hit identity.
- **Never re‑compute** if a cache hit exists with matching keys.
- Clear **only** the affected subcache when changing model id, crop, or template thresholds.

---

### 9) Not fine‑tuning, not RAG (FAQ)

- **Backbone**: ESM2 is **frozen** (no gradient updates). This is closer to **word2vec/GloVe‑style feature extraction**, but **contextual and per‑residue** (hidden states depend on full sequence).
- **PDB/mmCIF** is used **only** to build **labels/masks** (contact map at 8 Å), **not** to “decode” embeddings.
- **RAG?** No. We don’t retrieve text to augment prompts; we optionally add **numeric priors** (from homolog structures) into the contact head.

---

### 10) File map (excerpt)

```
configs/
  rescontact.yaml
scripts/
  train.py                 # train (BCEWithLogits)
  eval.py                  # eval (PR/ROC/F1; optional P@L)
  build_baseline.py        # PSI baseline
  monitor_eval.py          # PSI per split (batch-only)
  retrieve_homologs.py     # MMseqs2 (server-only) homolog retrieval
  build_template_priors.py # fetch PDB/AFDB, map & cache prior channels
src/rescontact/
  api/server.py            # /predict, /visualize (batch PSI = future)
  data/loader.py           # PDB/mmCIF parsing; masks & 8Å labels
  features/embedding.py    # ESM2 embedding cache (frozen backbone)
  model/head.py            # Linear→ReLU→Dropout→Bilinear + distance-bias
  templates/               # homology subsystem (remote-only provider)
    mmseqs.py, mapping.py, features.py, fuse.py, template_db.py
  utils/metrics.py
.cache/rescontact/         # caches (embeddings, MSA, templates)
```

---

### 11) Roadmap / limitations

- **Docker & Cloud** deployment are **future work** (kept **out** of the 7‑day roadmap files).
- **Streaming PSI** endpoints are planned; current PSI is **batch‑only** via scripts.
- For higher accuracy, consider larger ESM2 models or enabling MSA on a beefier machine.
