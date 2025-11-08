# Protein Engineering Monorepo — ResContact ➜ ResIntNet ➜ SeqML

End‑to‑end, modular pipeline for **protein & enzyme engineering**:
from **sequence** (± structure) and experimental datasets ➜
**contacts/priors** ➜ **residue‑interaction graph & hotspot ranking** (distal mutation candidates)
➜ **sequence‑level efficacy modeling** (fine‑tuned LMs).

This repository is organized as a **monorepo** with three packages and thin orchestration under `pipelines/`.
Each package is installable & runnable on its own; outputs are cached to disk so stages can be resumed or reused.

---

## Table of Contents

- [Goals](#goals)
- [Directory Layout](#directory-layout)
- [Install](#install)
- [Data & Caches](#data--caches)
- [Stage A — ResContact (contacts, priors, features)](#stage-a--rescontact-contacts-priors-features)
- [Stage B — ResIntNet (graphs, PRS/memory, GNN ranking)](#stage-b--resintnet-graphs-prsmemory-gnn-ranking)
- [Stage C — SeqML (mutant efficacy / fitness modeling)](#stage-c--seqml-mutant-efficacy--fitness-modeling)
- [Pipelines (E2E orchestration)](#pipelines-e2e-orchestration)
- [Configs](#configs)
- [Tips (Mac M1/M2/M3, CUDA), Repro, Docker](#tips-mac-m1m2m3-cuda-repro-docker)
- [Licenses & Data](#licenses--data)

---

## Goals

- **ResContact**: build template‑guided distance/contact priors and per‑residue features from ESM2 and MSA; train a compact contact head.
- **ResIntNet**: turn contacts/priors into residue‑interaction graphs; compute **PRS/memory** centrality and **GNN** scores; blend/ensemble to rank **distal** mutation hotspots.
- **SeqML**: fine‑tune sequence models (e.g., T5/ESM) for **mutant efficacy/fitness** prediction around proposed residues; iterate with experiment‑in‑the‑loop.

Everything is designed to run locally first (**caches on disk**), and later be portable to a DW/Lakehouse or workflow engine.

---

## Directory Layout

```
.
├── README.md
├── archive/
│   └── msa_utils/                  # old scripts archived
├── configs/
│   ├── mappings/
│   │   └── generic_example.yaml    # id/path mapping example
│   └── pipeline.example.yaml       # E2E config template
├── docker/                         # (future) base images / compose
├── packages/
│   ├── rescontact/                 # Stage A
│   │   ├── README.md
│   │   ├── notebooks/              # exploratory notebooks
│   │   ├── pyproject.toml
│   │   ├── scripts/                # CLIs (no heavy logic)
│   │   │   ├── retrieve_homologs.py, run_msa_batch.py, build_template_priors.py, ...
│   │   └── src/rescontact/         # library code
│   │       ├── io/ features/ datasets/ models/ training/ api/ utils/
│   ├── resintnet/                  # Stage B
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── scripts/                # ingest + ranking CLIs
│   │   │   ├── ingest_mutations.py
│   │   │   └── rank_mutations.py
│   │   └── src/resintnet/
│   │       ├── ingest/ graph.py prs.py models/ rank.py
│   └── seqml/                      # Stage C
│       ├── README.md
│       ├── pyproject.toml
│       ├── scripts/
│       │   ├── prepare_mutants.py
│       │   └── train.py
│       └── src/seqml/
│           ├── mutgen.py train_t5_lora.py
└── pipelines/
    ├── README.md
    ├── e2e_propose_mutations.py
    ├── train_rescontact.py
    ├── train_resintnet.py
    └── train_seqml.py
```

---

## Install

Use **per‑package** editable installs (keeps deps isolated). Recommended Python **3.11**.

```bash
# From repo root
python -m venv .venv && source .venv/bin/activate

# Stage A
pip install -e packages/rescontact

# Stage B
pip install -e packages/resintnet

# Stage C
pip install -e packages/seqml
```

> Tip: avoid installing ColabFold/JAX in this env unless you really need local MMseqs2+JAX; the CLIs here default to the
> **remote MMseqs2 API** (`https://a3m.mmseqs.com`) for MSAs, which avoids heavy deps.

---

## Data & Caches

All stages write to local **caches** (NPZ/NPY/Parquet/CSV) so you can resume. Common places:

- `data/fasta/*.fa` — your queries
- `data/templates/priors/*.npz` — ResContact template priors (distance‑bin probabilities)
- `data/emb/esm2_*/*.npy` — ESM2 per‑residue embeddings
- `data/msas/*.a3m` and `data/msa_features/*.npz` — MSA raw + summarized features
- `data/resintnet/*` — graphs, PRS and blended scores
- `data/seqml/*` — mutant candidates, train data, checkpoints

Environment knobs used by the scripts:

```bash
export RESCONTACT_TEMPLATE_DIR=data/templates
export CUDA_VISIBLE_DEVICES=0   # if CUDA available
```

---

## Stage A — ResContact (contacts, priors, features)

**Input**: FASTA(s), optional local PDBs.  
**Output**: per‑query NPZ/NPY features and contact/priors artifacts for downstream stages.

### A.1 Retrieve homologs (templates)

```bash
python packages/rescontact/scripts/retrieve_homologs.py \
  --fasta data/fasta/10_subset.fa \
  --server-url https://a3m.mmseqs.com \
  --max-hits 16 \
  --want-templates \
  --qps 0.2 --inter-job-sleep 2 --max-retries 8 --timeout 1800 \
  --flush-every 1 \
  --out data/templates/mmseqs_hits.json
```

### A.2 Build **template priors** (distance‑bin histograms)

Writes one NPZ per query with:
`priors (L,L,B)`, `bins`, `mask (L,L)`, `meta`.

```bash
export RESCONTACT_TEMPLATE_DIR=data/templates

python packages/rescontact/scripts/build_template_priors.py \
  --hits data/templates/mmseqs_hits.json \
  --pdb-root data/pdb/train data/pdb/test \
  --out-dir "$RESCONTACT_TEMPLATE_DIR/priors" \
  --structure-source "pdb,afdb" \
  --max-hits-per-query 8 \
  --max-downloads-per-run 50
```

### A.3 ESM2 embeddings

```bash
python packages/rescontact/scripts/embed_esm2.py \
  --fasta data/fasta/10_subset.fa \
  --out-dir data/emb/esm2_t12 \
  --model esm2_t12_35M_UR50D    # tiny, works on CPU/MPS
```

### A.4 Remote MSA + features

```bash
# Fetch A3M (remote MMseqs2 API)
python packages/rescontact/scripts/run_msa_batch.py \
  --fasta data/fasta/10_subset.fa \
  --msa-out-dir data/msas \
  --server-url https://a3m.mmseqs.com \
  --db uniref \
  --qps 0.15

# Summarize MSA ➜ features (PSSM/PSFM/MI-lite)
python packages/rescontact/scripts/build_msa_features.py \
  --msa-dir data/msas \
  --esm-emb-dir data/emb/esm2_t12 \
  --out-dir data/msa_features \
  --float16
```

### A.5 (Optional) Concatenate ESM2 + MSA features

```bash
python packages/rescontact/scripts/concat_esm2_msa.py \
  --esm-dir data/emb/esm2_t12 \
  --msa-dir data/msa_features \
  --out-dir data/emb/esm2_t12_plus_msa \
  --float16 \
  --mode pad \
  --include-depth \
  --verbose
```

### A.6 Train/eval the contact head

```bash
# Train
python packages/rescontact/scripts/train.py \
  --fasta data/fasta/10_subset.fa \
  --priors-dir data/templates/priors \
  --emb-dir data/emb/esm2_t12_plus_msa \
  --out-dir data/rescontact_runs/run1 \
  --epochs 20 --lr 3e-4 --batch 2

# Evaluate/monitor
python packages/rescontact/scripts/eval.py \
  --run-dir data/rescontact_runs/run1
```

---

## Stage B — ResIntNet (graphs, PRS/memory, GNN ranking)

**Input**: ResContact outputs (priors, contact probs, embeddings), optional PDB.  
**Output**: ranked residue hotspots (CSV) + optional GNN checkpoints.

**What it does**
- Build **residue‑interaction graph** (`graph.py`): nodes = residues; edges = contacts (Cα ≤ τ) or from top‑K prior probs.
- Node feats = ESM2 (+ MSA summaries, physchem if available); edge feats = binned distances, 1/r, template agreement, prior weight.
- **PRS/Memory** (`prs.py`): compute perturbation‑response centrality vector per residue.
- **GNN** (`models/sage.py`): GraphSAGE/GAT, predicts per‑residue propensity.
- **Blend** (`rank.py`): `final = α·sigmoid(GNN) + (1−α)·norm(PRS)`.

### B.1 (Optional) Ingest external mutation datasets

Use adapters in `src/resintnet/ingest/adapters/` to normalize curated mutation sets
(e.g., **D3DistalMutation**). Ensure licensing permits ML use.

```bash
# Example: D3Distal CSV -> normalized parquet
python packages/resintnet/scripts/ingest_mutations.py \
  --adapter d3distal \
  --input data/mutations/D3DistalMutation.csv \
  --out data/mutations/normalized.parquet \
  --mapping configs/mappings/generic_example.yaml
```

You can add more sources by implementing a small adapter under `adapters/` and registering it.

### B.2 Rank residues (PRS + optional GNN)

```bash
python packages/resintnet/scripts/rank_mutations.py \
  --fasta data/fasta/10_subset.fa \
  --priors-dir data/templates/priors \
  --emb-dir data/emb/esm2_t12_plus_msa \
  --out-dir data/resintnet/scores \
  --alpha 0.5 \
  --contact-threshold 8.0 \
  --topk-priors 400 \
  --save-graphs
```

This writes per‑protein CSVs with `residue_index`, `score_prs`, `score_gnn` (if trained), and `score_blend`.

> **Training the GNN**: use `pipelines/train_resintnet.py` for a minimal example that samples nodes/graphs from your proteins,
> optionally supervised by curated mutation labels (if available).

---

## Stage C — SeqML (mutant efficacy / fitness modeling)

**Input**: hotspot residues (from ResIntNet) + wild‑type sequences (+ experimental datasets if training).  
**Output**: fine‑tuned model & predictions for small mutational neighborhoods (1–3 AA changes) around hotspots.

### C.1 Generate candidate mutants

```bash
python packages/seqml/scripts/prepare_mutants.py \
  --fasta data/fasta/10_subset.fa \
  --hotspots data/resintnet/scores/*.csv \
  --out data/seqml/candidates.csv \
  --neighbors 1 \
  --per-residue 20
```

### C.2 Train or infer with a sequence model

```bash
# Example: LoRA fine‑tune a small T5 on curated fitness (if available)
python packages/seqml/scripts/train.py \
  --train data/seqml/train.csv \
  --val   data/seqml/val.csv \
  --out   data/seqml/t5_lora_run1

# Or run zero/few‑shot inference (see package README)
```

---

## Pipelines (E2E orchestration)

Use the thin scripts in `pipelines/` to glue stages together. They consume a single YAML.

```bash
python pipelines/e2e_propose_mutations.py --config configs/pipeline.example.yaml

# Or train stages individually
python pipelines/train_rescontact.py  --config configs/pipeline.example.yaml
python pipelines/train_resintnet.py   --config configs/pipeline.example.yaml
python pipelines/train_seqml.py       --config configs/pipeline.example.yaml
```

### Example `configs/pipeline.example.yaml`

```yaml
# Minimal example; tune paths & knobs to your setup.
data:
  fasta: data/fasta/10_subset.fa
  pdb_roots: [data/pdb/train, data/pdb/test]
  workdir: data/

mmseqs:
  server_url: https://a3m.mmseqs.com
  db: uniref
  qps: 0.15
  max_hits: 16
  want_templates: true

rescontact:
  max_hits_per_query: 8
  structure_source: [pdb, afdb]
  emb_model: esm2_t12_35M_UR50D
  use_msa: true
  concat_mode: pad
  include_depth: true

resintnet:
  contact_threshold: 8.0
  topk_priors: 400
  alpha_blend: 0.5
  save_graphs: true
  # gnn_training: optional block if you have labels
  # gnn_training:
  #   epochs: 50
  #   lr: 1.0e-3

seqml:
  neighbors: 1
  per_residue: 20

runtime:
  device: auto    # cuda|mps|cpu
  num_workers: 2
  float16: true
```

---

## Tips (Mac M1/M2/M3, CUDA), Repro, Docker

- Apple Silicon works well with **PyTorch MPS** for ESM2 embedding and small models. Keep batch sizes small.
- Remote MSA avoids heavy local installs. If you need local MMseqs2, isolate it in a separate env/docker image.
- Use **`pip install -e`** per package to keep dependency islands small.
- Dockerfiles in `docker/` (coming) will pin OS/driver combos for reproducibility.

---

## Licenses & Data

- Code is under your repo’s license (TBD). Third‑party libraries keep their own licenses.
- **External datasets** (e.g., D3DistalMutation) may have **usage restrictions**. Confirm license terms before training or redistribution.
- If you add new dataset adapters, document the source and license in `packages/resintnet/src/resintnet/ingest/README.md`.

---

## What’s next

- Add CI in `.github/workflows/` to lint & run a tiny smoke test per package.
- Publish minimal Docker images for each package.
- Optional: move persistent artifacts to cloud object storage and add a DW schema when you outgrow local caches.

Happy hacking! ✨
