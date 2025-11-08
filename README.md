# protein-engineering — end‑to‑end sequence → structure/contacts → hotspot ranking → mutant efficacy

This monorepo provides an **end‑to‑end, laptop‑friendly** pipeline for protein & enzyme engineering.
Given **sequence(s)** and (optionally) **experimental data**, we:
1) Build **structure‑aware contact/distance priors** (ResContact) from templates + ESM2 + MSA.
2) Construct a **residue‑interaction network** (ResIntNet) and rank **distal mutation hotspots** by blending **GNN scores** with **memory/PRS centrality**.
3) (Optional) **SeqML** fine‑tunes sequence models (e.g., T5/ESM) to predict mutant **efficacy/fitness** around those hotspots.

> Designed to run on a MacBook Air M‑series (8 GB) using small models, float16 features, remote MSAs, and shallow GNNs.


---

## A) Repository layout (monorepo)

```
protein-engineering/
├── README.md
├── archive/
│   └── msa_utils/                # legacy/experimental scripts kept for reference
├── configs/
│   └── mappings/
│       └── generic_example.yaml  # example mapping config for datasets/IDs
├── docker/                       # base images or compose (optional)
├── packages/
│   ├── rescontact/               # contact/distance priors (templates, ESM2, MSA)
│   │   ├── README.md
│   │   ├── notebooks/
│   │   │   └── res_contact_workflow*.ipynb
│   │   ├── pyproject.toml
│   │   ├── scripts/
│   │   │   ├── embed_esm2.py
│   │   │   ├── run_msa_batch.py
│   │   │   ├── build_msa_features.py
│   │   │   ├── retrieve_homologs.py
│   │   │   ├── build_template_priors.py
│   │   │   ├── fuse_priors.py
│   │   │   ├── train.py (baseline contact model)
│   │   │   └── eval.py
│   │   └── src/rescontact/
│   │       ├── datasets/         # dataset + loader
│   │       ├── features/         # esm.py, msa.py, pair.py, fuse.py
│   │       ├── io/               # mmseqs, pdb mapping, template_db
│   │       ├── models/           # contact_net, heads, bilinear_scorer
│   │       ├── training/         # metrics, psi (Population Stability Index), train
│   │       └── api/              # (optional) simple server
│   ├── resintnet/                # residue‑interaction network + mutation ranking
│   │   ├── README.md
│   │   ├── notebook/
│   │   │   ├── prot_rin_gnn.ipynb   # GNN training tutorial
│   │   │   └── prot_rin_mem.ipynb   # PRS/Memory tutorial
│   │   ├── pyproject.toml
│   │   ├── scripts/
│   │   │   ├── ingest_mutations.py  # normalize external mutation datasets
│   │   │   └── rank_mutations.py    # build graph, (optionally load ckpt), blend with PRS
│   │   └── src/resintnet/
│   │       ├── graph.py
│   │       ├── prs.py
│   │       ├── rank.py
│   │       └── ingest/
│   │           ├── base.py
│   │           ├── utils.py
│   │           └── adapters/         # plug‑ins for curated datasets
│   │               ├── d3distal.py
│   │               └── generic_csv.py
│   └── seqml/                       # sequence‑level modeling (mutant efficacy/fitness)
│       ├── README.md
│       ├── notebook/
│       │   ├── PT5_xl_ACT.ipynb
│       │   └── PT5_xl_GB1.ipynb
│       ├── prot_api_flask/
│       ├── pyproject.toml
│       ├── scripts/
│       │   ├── prepare_mutants.py   # enumerate mutants around hotspots
│       │   └── train.py             # fine‑tune/fit simple seq model
│       └── src/seqml/
│           ├── mutgen.py
│           └── train_t5_lora.py     # example LoRA fine‑tune (small)
└── pipelines/                     # thin orchestration (glue only)
```

**Why this layout?**
- Each package is installable (`pip install -e packages/<name>`).
- Scripts are **thin** CLIs that call into `src/…` so you can reuse them.
- `pipelines/` contains only orchestration flows (no heavy logic).


---

## B) End‑to‑end flow

### 1) ResContact — embeddings, MSAs, and template‑based distance priors

**Inputs**: FASTA, optional local PDB cache.  
**Outputs (per query ID)**:
- `data/emb/esm2_t12/{ID}.esm2.npy` — ESM2 per‑residue embeddings (L×C)
- `data/msas/{ID}.a3m` + `.tgz` — remote MMseqs A3M (no local MMseqs install)
- `data/msa_features/{ID}.npz` — compact MSA features (`X`(L×F), `depth`, `meta`)
- `data/templates/priors/{ID}.npz` — `(priors[L×L×B], bins, mask, meta)`

**Minimal commands:**

```bash
# 1a) ESM2 embeddings (tiny model fits Mac M‑series)
python packages/rescontact/scripts/embed_esm2.py \
  --fasta data/fasta/10_subset.fa \
  --out-dir data/emb/esm2_t12 \
  --model esm2_t12_35M_UR50D

# 1b) Remote MSA via MMseqs API (rate‑limited)
python packages/rescontact/scripts/run_msa_batch.py \
  --fasta data/fasta/10_subset.fa \
  --msa-out-dir data/msas \
  --server-url https://a3m.mmseqs.com \
  --db uniref --qps 0.15

# 1c) MSA → compact features (depth, PSSM/PSFM, MI/APC summaries)
python packages/rescontact/scripts/build_msa_features.py \
  --msa-dir data/msas \
  --esm-emb-dir data/emb/esm2_t12 \
  --out-dir data/msa_features \
  --float16

# 1d) Template priors (PDB/AFDB) → (L×L×B) distance histograms
export RESCONTACT_TEMPLATE_DIR=data/templates
python packages/rescontact/scripts/build_template_priors.py \
  --hits data/templates/mmseqs_hits.json \
  --pdb-root data/pdb/train data/pdb/test \
  --out-dir "$RESCONTACT_TEMPLATE_DIR/priors" \
  --structure-source "pdb,afdb" \
  --max-hits-per-query 8 \
  --max-downloads-per-run 50
```

**What’s inside `{ID}.npz` priors?**
- `priors`: `(L, L, B)` prob. over distance bins from selected templates
- `bins`: bin edges used to build histograms (keep for training)
- `mask`: `(L, L)` indicating positions with non‑zero prior
- `meta`: JSON string with `{query_id, L, templates_used, …}`

**Tips for 8 GB laptops**
- Prefer `esm2_t12_35M_UR50D` (C=480); set `--float16` where supported.
- Use `--qps` ≤ 0.2 for MMseqs; MSAs are cached in `data/msas`.
- Cap templates per query (`--max-hits-per-query 8`) to keep priors small.


### 2) ResIntNet — graph construction, GNN ranking, PRS/Memory blending

We build a **residue‑interaction graph** per protein and score residues.

**Nodes**: residues 1..L with features  
`x_node = concat([ESM2_i, MSA_i, optional physchem_i])`

**Edges**: either
- Geometry edges (CA distance<thr, e.g., 8Å), **or**
- Top‑K neighbors from priors (e.g., `K=16` / residue).

**Edge attributes**: `[1/dist, one_hot(dist_bin), template_agreement, …]`

**Labels (optional)**: if you have curated distal‑mutation data, convert to **residue‑level labels** (see Section C).

**Workflows**:

- **Unsupervised (graph + PRS + heuristic blend)**  
  Use `rank_mutations.py` to build graphs from ESM2/MSA/Priors and compute PRS; blend with a shallow heuristic if no trained GNN is available.

- **Supervised (recommended if labels exist)**  
  Train a small GNN (GraphSAGE/GAT) using the notebooks (`notebook/prot_rin_gnn.ipynb`) or your own training loop. Then **blend** with PRS using `rank_mutations.py`.

**Commands (examples):**

```bash
# 2a) Normalize external datasets into a single CSV of residue labels
python packages/resintnet/scripts/ingest_mutations.py \
  --source d3distal \
  --input data/raw/D3DistalMutation.csv \
  --out-csv data/supervision/distal_mutations.csv \
  --mapping-config configs/mappings/generic_example.yaml

# 2b) Rank residues (uses GNN ckpt if provided; otherwise PRS‑only / heuristic)
python packages/resintnet/scripts/rank_mutations.py \
  --esm-dir data/emb/esm2_t12 \
  --msa-dir data/msa_features \
  --priors-dir data/templates/priors \
  --labels-csv data/supervision/distal_mutations.csv \
  --graphs-out data/graphs \
  --ckpt runs/gnn/best.ckpt \        # optional (if you trained a GNN)
  --prs-alpha 0.4 \
  --out-dir outputs/ranks
```

**Outputs**: `outputs/ranks/{protein_id}.csv` with
`residue_index_seq, gnn_score (if any), prs, final_score` (sorted desc).

**Blend formula**:  
`final = α · sigmoid(gnn_logits) + (1−α) · prs_norm`, with `α∈[0.3,0.7]`.


### 3) SeqML — mutant generation & efficacy/fitness modeling (optional)

Use the ranked residues to generate small mutational neighborhoods and fine‑tune lightweight models to predict **efficacy/fitness** (depending on your assay).

```bash
# 3a) Enumerate single/double mutants at top‑K residues
python packages/seqml/scripts/prepare_mutants.py \
  --fasta data/fasta/10_subset.fa \
  --hotspot-csv outputs/ranks/106M_A.csv \
  --k 10 --max-muts-per-res 5 \
  --out data/mutants/106M_A_candidates.csv

# 3b) Train a small model (e.g., token/char‑CNN or T5‑LoRA in notebooks)
python packages/seqml/scripts/train.py \
  --train-csv data/mutants/train.csv \
  --val-csv   data/mutants/val.csv \
  --out-dir   runs/seqml/baseline
```

For richer models (T5‑LoRA), see `packages/seqml/notebook/*`.


---

## C) Supervision with curated distal‑mutation datasets

**Goal**: leverage curated mutation datasets (e.g., distal/allosteric mutations from the literature) to supervise the GNN.

> ⚠️ Always check licensing/terms. Do not store third‑party raw data in the repo; use an **ingest adapter** + mapping config.

### C.1 Normalized CSV schema (residue‑level)

`data/supervision/distal_mutations.csv`

| column              | type  | notes |
|---------------------|-------|-------|
| `protein_id`        | str   | e.g., `1ABC_A` or your query ID |
| `chain_id`          | str   | optional if encoded in protein_id |
| `residue_index_seq` | int   | **sequence index** (1‑based; aligns to FASTA/ESM2) |
| `label`             | int   | 1 = distal hotspot, 0 = non‑hotspot |
| `effect_size`       | float | optional (Δactivity, Δstability, …) |
| `evidence`          | str   | optional citation/source |
| `split`             | str   | `train` / `val` / `test` |

If a source uses **PDB numbering**, map to **sequence indices** using your mapping util (see `rescontact.io` modules).

### C.2 Ingest adapters

`packages/resintnet/src/resintnet/ingest/adapters/` contains pluggable adapters.  
Example usage:

```bash
python packages/resintnet/scripts/ingest_mutations.py \
  --source d3distal \
  --input /path/to/D3DistalMutation.csv \
  --out-csv data/supervision/distal_mutations.csv \
  --mapping-config configs/mappings/generic_example.yaml
```

To add new datasets, implement a new adapter `mydataset.py` and register it in the CLI.


---

## D) Metrics & evaluation

- **Per‑protein**: AUROC, AUPRC, Hit@K (K=5,10,20), Enrichment@K.
- **Global**: macro/micro AUPRC across proteins.
- **Stability (PSI)**: `packages/rescontact/scripts/psi_report.py` for Population Stability Index if your dataset drift monitoring needs it.


---

## E) Installation & environments

Create a clean env (Python ≥3.10). For Apple Silicon, PyTorch with MPS is recommended.

```bash
# Example: venv + pip (or conda/mamba if you prefer)
python -m venv .venv && source .venv/bin/activate

# Install packages in editable mode (start with rescontact; others as needed)
pip install -e packages/rescontact
pip install -e packages/resintnet
pip install -e packages/seqml

# Torch (CPU or MPS on Apple Silicon)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # choose a wheel matching your setup
```

> **MSA note**: the repo uses **remote MMseqs** (`https://a3m.mmseqs.com`). You do **not** need ColabFold locally for MSAs. Set `--qps` to respect rate limits.


---

## F) Configs

Example mapping config (`configs/mappings/generic_example.yaml`):

```yaml
# Maps external dataset identifiers to your internal protein IDs and indices
sequence_to_pdb_map:
  "106M_A": { pdb_id: "1a9w", chain: "F" }
  "109L_A": { pdb_id: "1p2r", chain: "A" }

indexing:
  scheme: "sequence"     # PDB | sequence
  offset: 1              # 1‑based sequence indices
```


---

## G) Orchestration (pipelines)

`pipelines/` holds **thin** entrypoints (Airflow/Prefect/CLI) that call the package CLIs. A future `e2e_propose_mutations.py` will chain:

1. ResContact: ESM2 → MSA → priors  
2. ResIntNet: graph → (optional GNN) → PRS blend → ranked residues  
3. SeqML: enumerate mutants → score/fit

> Keep orchestration stateless; cache **only** in `data/` (git‑ignored).


---

## H) Repro & logging

- Seed everything (`numpy`, `torch`).
- Log configs to `runs/…`.
- Split **by protein**, not by residue, to avoid leakage.


---

## I) Licensing & data

- This repo is **code‑only**. External datasets (e.g., distal/allosteric mutations) **must not** be checked in. Use ingest scripts and keep raw sources in `data/raw/` (git‑ignored).
- Check the **license/terms** of external datasets to ensure training use is permitted.


---

## J) Quick start (copy‑paste)

```bash
# 0) Prepare
mkdir -p data/fasta data/emb/esm2_t12 data/msas data/msa_features data/templates/priors

# 1) ResContact
python packages/rescontact/scripts/embed_esm2.py \
  --fasta data/fasta/10_subset.fa \
  --out-dir data/emb/esm2_t12 \
  --model esm2_t12_35M_UR50D

python packages/rescontact/scripts/run_msa_batch.py \
  --fasta data/fasta/10_subset.fa \
  --msa-out-dir data/msas \
  --server-url https://a3m.mmseqs.com \
  --db uniref --qps 0.15

python packages/rescontact/scripts/build_msa_features.py \
  --msa-dir data/msas \
  --esm-emb-dir data/emb/esm2_t12 \
  --out-dir data/msa_features --float16

export RESCONTACT_TEMPLATE_DIR=data/templates
python packages/rescontact/scripts/build_template_priors.py \
  --hits data/templates/mmseqs_hits.json \
  --pdb-root data/pdb/train data/pdb/test \
  --out-dir "$RESCONTACT_TEMPLATE_DIR/priors" \
  --structure-source "pdb,afdb" \
  --max-hits-per-query 8 --max-downloads-per-run 50

# 2) (Optional) Supervision ingest
python packages/resintnet/scripts/ingest_mutations.py \
  --source d3distal \
  --input data/raw/D3DistalMutation.csv \
  --out-csv data/supervision/distal_mutations.csv \
  --mapping-config configs/mappings/generic_example.yaml

# 3) Ranking (PRS + optional GNN)
python packages/resintnet/scripts/rank_mutations.py \
  --esm-dir data/emb/esm2_t12 \
  --msa-dir data/msa_features \
  --priors-dir data/templates/priors \
  --labels-csv data/supervision/distal_mutations.csv \
  --prs-alpha 0.4 \
  --out-dir outputs/ranks
```

Happy hacking! 🧪🧬
