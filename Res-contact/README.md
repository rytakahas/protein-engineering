#### Res-Contact — ESM2-based Protein Contact Prediction (Laptop‑friendly)

A lightweight, cache-first pipeline for residue–residue **contact prediction** that:
- Reuses a **frozen ESM2 backbone** (feature extractor) and trains a tiny **bilinear head**.
- Builds ground-truth contact labels from PDB/mmCIF (**Cα–Cα ≤ 8.0 Å**).
- Optionally ingests **homology templates** via a **server-only MMseqs2** path (no local UniRef DB).
- Supports **optional 1-D MSA features** (+21 dims) when available.
- Provides **batch PSI monitoring** to track score distribution drift (train→val/test).
- Ships a minimal **FastAPI** server for inference/visualization (live PSI = future work).

> **Designed for an 8‑GB MacBook Air (M3)**: cache-first, compute-light, and optional extras that degrade gracefully.

---

##### 1) What’s included

**Core training/eval**
- `scripts/train.py` — Train head with BCEWithLogits on the strict upper triangle (i<j)
- `scripts/eval.py` — Evaluate metrics (PR‑AUC, ROC‑AUC, F1; optional P@L for context)

**Monitoring (batch only)**
- `scripts/build_baseline.py` — Build PSI **baseline** (quantile bins on train)
- `scripts/monitor_eval.py` — Compute PSI & histograms per split using the baseline

**Homology templates (server-only MMseqs2; cache-first)**
- `scripts/retrieve_homologs.py` — Query a **remote** MMseqs2 server and cache hits as JSON
- `scripts/build_template_priors.py` — Fetch **PDB/AFDB** structures for top hits (cached), map residues, and build **contact priors**
- `src/rescontact/templates/mmseqs.py` — Remote MMseqs2 client (probes `/easy-search | /api/search | /search`), **no local DB**
- `src/rescontact/templates/template_db.py` — Minimal HTTP fetchers for **RCSB PDB** & **AlphaFold DB**, with on-disk cache
- `src/rescontact/templates/mapping.py` — Global alignment (query↔hit) and residue index mapping
- `src/rescontact/templates/features.py` — Build prior channels (contact map / distogram) for the query length
- `src/rescontact/templates/fuse.py` — Lightweight **logit‑blend** fusion (keeps the ESM2 head unchanged)

**Features & model**
- `src/rescontact/features/embedding.py` — **ESM2** embeddings cache (frozen backbone; contextual per-residue vectors)
- `src/rescontact/models/bilinear_scorer.py` + `src/rescontact/models/contact_net.py` — small head (Linear→ReLU→Dropout→Bilinear + distance‑bias)
- `src/rescontact/data/dataset.py`, `src/rescontact/data/pdb_utils.py` — PDB/mmCIF parsing, masks, labels @ 8 Å
- `src/rescontact/utils/metrics.py`, `src/rescontact/utils/psi.py`, `src/rescontact/utils/train.py`

**API**
- `src/rescontact/api/server.py` — FastAPI `/predict` and `/visualize` (PSI endpoints planned for a future version)

**Configs**
- `configs/rescontact.yaml` — default (ESM‑only, optional MSA)
- `configs/rescontact.server.yaml` — example enabling **server‑only MMseqs2 templates** + logit blending

**Optional tuning**
- `optuna_sweep.py` — quick & coarse hyperparameter sweep (hidden size, lr, dropout, decision threshold)

> **Containerization** is **future work** and intentionally **not** in the roadmap files.


---

##### 2) File tree (key parts)

```
Res-contact/
├─ README.md
├─ configs/
│  ├─ rescontact.yaml                 # default config (ESM-only w/ optional MSA)
│  └─ rescontact.server.yaml          # server-only MMseqs2 + template fusion (example)
├─ data/
│  ├─ fasta/                          # input sequences (FASTA)
│  ├─ msa/                            # optional 1-D MSA features cache
│  └─ pdb/{train,test}/               # PDB/mmCIF structures for labels
├─ scripts/
│  ├─ train.py                        # train (full-grid head) — BCEWithLogits
│  ├─ eval.py                         # eval & metrics (PR/ROC/F1; optional P@L)
│  ├─ build_baseline.py               # PSI baseline (quantile bins on train)
│  ├─ monitor_eval.py                 # compute PSI & histos per split (batch)
│  ├─ retrieve_homologs.py            # MMseqs2 remote retrieval (server-only; cached)
│  └─ build_template_priors.py        # fetch structures + build priors (cached)
├─ src/rescontact/
│  ├─ api/server.py                   # FastAPI app (/predict, /visualize); live PSI = future
│  ├─ data/{dataset.py,pdb_utils.py}  # parsing, masks, labels
│  ├─ features/{embedding.py,msa.py,pair_features.py}
│  ├─ models/{bilinear_scorer.py,contact_net.py}
│  ├─ templates/                      # homology templates subsystem (server-only)
│  │  ├─ mmseqs.py                    # remote MMseqs2 client (no local DB)
│  │  ├─ template_db.py               # PDB/AFDB HTTP fetchers + cache
│  │  ├─ mapping.py                   # query↔hit residue mapping
│  │  ├─ features.py                  # build prior channels
│  │  └─ fuse.py                      # logit blending
│  └─ utils/{metrics.py,psi.py,train.py}
├─ tests/
│  ├─ test_pdb_utils.py
│  ├─ test_bilinear_scorer.py
│  ├─ test_pair_features.py
│  ├─ test_msa_providers_mock.py
│  └─ test_train_smoke.py
└─ .cache/rescontact/                 # embedding, hits, structures, priors
```

---

##### 3) Setup

**Python**: 3.10–3.11 recommended  
**Hardware**: 8‑GB Mac (M3) OK; uses **MPS** if available

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Mac MPS stability tips
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export MAX_LEN_PER_CHAIN=600      # tune for memory; 600 works on 8‑GB
```

---

##### 4) Ground truth & embeddings (ESM2 is frozen)

- **Labels**: build binary contacts from PDB/mmCIF (**Cα–Cα ≤ 8.0 Å**) on the **strict upper triangle** (i<j).
- **ESM2**: `facebook/esm2_t6_8M_UR50D` produces contextual **per‑residue vectors** (L×320). These are cached to disk.
- **MSA (optional)**: if present, append +21 dims (AA frequencies + entropy). If missing, zeros are used (shape‑stable).

> This is **not fine‑tuning** and **not RAG** — ESM2 is a **frozen feature extractor**. PDB is used for **labels**, not for embedding.


---

# 5) Training & evaluation

**Train**
```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/rescontact.yaml \
  --epochs 20 --batch-size 1
```

**Eval**
```bash
PYTHONPATH=src python scripts/eval.py \
  --config configs/rescontact.yaml \
  --ckpt checkpoints/model_best.pt \
  --split test --max_test_examples 500
```

**Metrics**: PR‑AUC, ROC‑AUC, F1 (masked upper triangle).  
`P@L` can be printed as context but, without long‑range filtering, it may be dominated by near‑diagonal pairs.


---

##### 6) Monitoring (PSI drift) — batch only

**Build baseline (once)**
```bash
PYTHONPATH=src python scripts/build_baseline.py \
  --config configs/rescontact.yaml \
  --out monitor/baseline.json \
  --max_examples 200
```

**Compute PSI on splits using the baseline**
```bash
PYTHONPATH=src python scripts/monitor_eval.py \
  --config configs/rescontact.yaml \
  --ckpt checkpoints/model_best.pt \
  --baseline monitor/baseline.json \
  --split val --split test
```

Outputs under `reports/`:
- `psi_<split>_<ts>.json` (value, category, proportions, meta)
- `score_<split>_<ts>.png` (probability histogram)
- `length_<split>_<ts>.png` (sequence length distribution)
- `sep_<split>_<ts>.png` (|i−j| separation distribution)

Thresholds (configurable): **≤ 0.10 stable**, **0.10–0.25 watch**, **> 0.25 drift**.  
> Live PSI endpoints for the server are **future work**.


---

##### 7) Homology templates — **server‑only MMseqs2** (no local DB)

This path **does not** require any local UniRef/Uniprot databases.
Everything is **download-once** and **cache‑first**.

**Env**
```bash
export MMSEQS_SERVER_URL="https://your-mmseqs-server"   # the remote MMseqs2 endpoint
```

**1) Retrieve homologs (cached JSON hits)**
```bash
PYTHONPATH=src python scripts/retrieve_homologs.py \
  --fasta data/fasta/demo.fasta \
  --out data/templates/mmseqs_hits.json \
  --server-url "$MMSEQS_SERVER_URL" \
  --db uniref90 --max-hits 8 --min-ident 0.30 --min-cov 0.60
```

**2) Build template priors (cached structures)**
```bash
PYTHONPATH=src python scripts/build_template_priors.py \
  --hits data/templates/mmseqs_hits.json \
  --query-fasta data/fasta/demo.fasta \
  --out-dir .cache/rescontact/templates/priors \
  --max-structures 4
```

This fetches only the **few** PDB/AFDB files needed (HTTP) and caches them under `.cache/rescontact/templates/`.  
Mapping aligns query↔hit residues; priors are shaped to **query length L**.


###### Fusion (config only; model unchanged)

Use **logit blending** to inject the template prior into the head’s logits.

```yaml
# configs/rescontact.server.yaml  (example)
templates:
  enabled: true
  mmseqs:
    server_url: ${env:MMSEQS_SERVER_URL}
    db: "uniref90"
    max_hits: 8
    min_ident: 0.30
    min_cov: 0.60
  prior:
    type: "contact"       # or "distogram" (if enabled)
    max_structures: 4
  fusion:
    mode: "logit_blend"
    alpha: 0.25           # 0 = ignore prior; 1 = prior dominates
```

Train/Eval will pick up priors if present in cache and `templates.enabled: true`.  
If no priors exist, it **silently falls back** to ESM‑only.


---

##### 8) FastAPI (minimal)

Start the server:
```bash
PYTHONPATH=src uvicorn src.rescontact.api.server:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /predict` — JSON input `{ "sequence": "...", "threshold": 0.5 }` → returns scores/pairs
- `POST /visualize` — returns a base64 heatmap (probabilities)
- Health/metrics: basic; **live PSI** is **future work**

> **Containerization / Cloud deploy** is **future work** (intentionally **not** in the roadmap).


---

##### 9) Config cheatsheet

```yaml
model:
  esm_model: "facebook/esm2_t6_8M_UR50D"
  embed_dim: 320
features:
  use_msa: false            # true if you have +21-dim 1D MSA features
data:
  pdb_root: "data/pdb"
  max_len_per_chain: ${env:MAX_LEN_PER_CHAIN, 600}
train:
  epochs: 20
  batch_size: 1
  lr: 1.5e-3
  dropout_p: 0.1
eval:
  threshold: 0.5
templates:                   # server-only MMseqs2 (optional)
  enabled: false             # set true to enable
  mmseqs:
    server_url: ${env:MMSEQS_SERVER_URL, ""}
    db: "uniref90"
    max_hits: 8
    min_ident: 0.3
    min_cov: 0.6
  prior:
    type: "contact"
    max_structures: 4
  fusion:
    mode: "logit_blend"
    alpha: 0.25
monitoring:
  psi:
    baseline_path: "monitor/baseline.json"
    thresholds: [0.10, 0.25]
```

---

##### 10) Tips & troubleshooting

- **MPS op gaps**: If you see `aten::triu_indices not implemented for MPS`, set:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- **ESM2 cache growth**: the embeddings are the largest artifacts. Clean old runs under `.cache/rescontact/` if space is tight.
- **Optuna/SQLAlchemy typing_extensions**: if you hit `TypeAliasType` import errors on macOS:
  ```bash
  pip install -U "typing_extensions>=4.12" "SQLAlchemy<2.0"
  ```
  or use the **in‑memory** study `--study sqlite:///:memory:` for quick local sweeps.
- **P@L ~1.0** without long‑range split is often meaningless (dominated by near‑diagonal). Prefer PR‑AUC / ROC‑AUC / F1 and optionally long‑range metrics.
- **Homology priors**: set a **min identity/coverage** you trust (e.g., 30/60%). Increase `alpha` only if priors look clean.


---

##### 11) Roadmap alignment (consistency note)

- `roadmap.txt` / `Roadmap.xlsx` **do not include containerization**; it’s labeled **future work** here.
- The **report** and **README** agree on: frozen ESM2, Cα–Cα ≤ 8 Å, optional MSA, **batch PSI only**, and **server‑only MMseqs2** for homology templates with cache-first design.
