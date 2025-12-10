# Protein Engineering Monorepo — ResContact → ResIntNet → SeqML

End-to-end, modular pipeline for **protein & enzyme engineering**:
from **sequence** (± structure) and experimental datasets →
**contacts/priors** → **residue-interaction graph & hotspot ranking** (distal mutation candidates)
→ **sequence-level efficacy modeling** (fine-tuned LMs).

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
- [What's next](#whats-next)

---

## Goals

- **rescontact**: build template-guided distance/contact priors and per-residue features from ESM2 and MSA; train a compact contact head.
- **resintnet**: turn contacts/priors into residue-interaction graphs; compute **PRS (Perturbation-Response Scanning / Signal) + memory-based** centrality and **GNN (Graph Neural Network)** scores; blend/ensemble to rank **distal** mutation hotspots.
- **seqml**: fine-tune sequence models (e.g., T5/ESM) for **mutant efficacy/fitness** prediction around proposed residues; iterate with experiment-in-the-loop.

Everything is designed to run locally first (**caches on disk**), and later be portable to a DW/Lakehouse or workflow engine.

---

## Directory Layout

```
.
├── README.md
├── archive/
│   └── msa_utils/
│       ├── Dockerfile
│       ├── README.md
│       └── run_deepmsa.sh
├── configs/
│   ├── mappings/
│   │   └── generic_example.yaml      # example for mapping/query metadata
│   └── pipeline.example.yaml         # E2E config template
├── docker/                           # (future) base images / compose
├── packages/
│   ├── rescontact/                   # Stage A
│   │   ├── README.md
│   │   ├── notebooks/
│   │   │   ├── res_contact_workflow.ipynb
│   │   │   ├── res_contact_workflow_opt.ipynb
│   │   │   ├── res_contact_workflow_opt2.ipynb
│   │   │   └── res_contact_workflow_opt_msa.ipynb
│   │   ├── pyproject.toml
│   │   ├── scripts/                  # CLIs (no heavy logic)
│   │   │   ├── retrieve_homologs.py
│   │   │   ├── run_msa_batch.py
│   │   │   ├── build_template_priors.py
│   │   │   ├── build_msa_features.py
│   │   │   ├── embed_esm2.py
│   │   │   ├── concat_esm2_msa.py
│   │   │   ├── fuse_priors.py
│   │   │   ├── train.py
│   │   │   ├── eval.py
│   │   │   └── psi_report.py
│   │   └── src/rescontact/           # library code
│   │       ├── api/ features/ io/ datasets/ models/ training/ utils/
│   ├── resintnet/                    # Stage B
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── scripts/
│   │   │   ├── ingest_mutations.py
│   │   │   └── rank_mutations.py
│   │   └── src/resintnet/
│   │       ├── graph.py prs.py rank.py
│   │       ├── models/
│   │       │   └── sage.py
│   │       └── ingest/
│   │           ├── base.py utils.py
│   │           └── adapters/
│   │               ├── d3distal.py
│   │               └── generic_csv.py
│   └── seqml/                        # Stage C
│       ├── README.md
│       ├── pyproject.toml
│       ├── scripts/
│       │   ├── prepare_mutants.py
│       │   └── train.py
│       └── src/seqml/
│           ├── mutgen.py
│           └── train_t5_lora.py
└── pipelines/
    ├── README.md
    ├── e2e_propose_mutations.py
    ├── train_rescontact.py
    ├── train_resintnet.py
    └── train_seqml.py
```

---

## Install

Use per-package editable installs (keeps deps isolated). Recommended Python 3.11.

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

**Tip:** avoid installing ColabFold/JAX in this env unless you really need local MMseqs2+JAX; the CLIs here default to the remote MMseqs2 API (https://a3m.mmseqs.com) for MSAs, which avoids heavy deps.

---

## Data & Caches

All stages write to local caches (NPZ/NPY/Parquet/CSV) so you can resume. Common places:

- `data/fasta/*.fa` — your queries
- `data/templates/priors/*.npz` — ResContact template priors (distance-bin probabilities)
- `data/emb/esm2_*/*.npy` — ESM2 per-residue embeddings
- `data/msas/*.a3m` and `data/msa_features/*.npz` — MSA raw + summarized features
- `data/resintnet/*` — graphs, PRS/memory and blended scores
- `data/seqml/*` — mutant candidates, train data, checkpoints

Environment knobs used by the scripts:

```bash
export RESCONTACT_TEMPLATE_DIR=data/templates
export CUDA_VISIBLE_DEVICES=0   # if CUDA available
```

---

## Stage A — ResContact (contacts, priors, features)

**Input:** FASTA(s), optional local PDBs.  
**Output:** per-query NPZ/NPY features and contact/priors artifacts for downstream stages.

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

### A.2 Build template priors (distance-bin histograms)

Writes one NPZ per query with: `priors` (L,L,B), `bins`, `mask` (L,L), `meta`.

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

# Summarize MSA → features (PSSM/PSFM/MI-lite)
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

**Input:** ResContact outputs (priors, contact probs, embeddings), optional PDB.  
**Output:** ranked residue hotspots (CSV) + optional GNN checkpoints.

### What it does

1. **Build residue-interaction graph** (`src/resintnet/graph.py`):
   - **Nodes** = residues.
   - **Node features** = ESM2 (+ MSA summaries; optional physicochemical features).
   - **Edges** from contacts (Cα ≤ τ Å) or top-K from priors.
   - **Edge attributes** include, e.g.:
     - distance bin features (one-hot / soft),
     - prior contact probability,
     - memory conductance C (see PRS/memory below),
     - 1 / r or RBF distance features,
     - template agreement / count.

2. **PRS / memory flow** (`src/resintnet/prs.py`):

   We use a simple, ENM-style perturbation–response model:

   - **PRS (Perturbation-Response Scanning / Signal):**  
     We apply a small virtual "kick" or load at one site (e.g. orthosteric pocket or interface) on the residue graph and solve a Laplacian-type system to estimate how strongly other residues respond. This yields a flow / centrality signal that has been widely used as a proxy for allosteric communication pathways in elastic-network models.

   - **Memory conductance C:**  
     We treat an edge-wise conductance as a "memory" channel. Edges that consistently carry strong PRS flow can be given higher C, while low-flow edges get lower C. In the simplest version (current code) this is a one-shot update based on PRS flux; in future work it can be iterated in the spirit of adaptive-network models (e.g. Bhattacharyya et al., PRL 129, 028101, 2022), where the network adapts to repeated loads and its topology "remembers" preferred communication routes.

   From this we compute:
   - **PRS-like centrality** per residue (how strongly each residue responds to orthosteric perturbation),
   - optional **edge-level "memory focus"** scores (which contacts appear on strong communication paths).

3. **GNN scoring** (`src/resintnet/models/sage.py`):
   - A Graph Neural Network (GNN) (GraphSAGE-style) consumes the node and edge features.
   - The memory conductance C is exposed as a dedicated edge feature and can also be used as a weight in message passing.
   - The GNN can be trained on weak labels (graph-level activity/stability aggregates) and/or curated mutation datasets. We also support auxiliary losses that encourage the GNN's internal gates and saliency to align with PRS-derived flows.

4. **Blend & rank** (`src/resintnet/rank.py`):

   Final per-residue score is a blend of physics-inspired and learned terms:

   ```
   score = α · σ( GNN(residue) )
         + β · norm(PRS_centrality)
         + γ · MemoryFocus
   ```

   with default α = 0.5, β = 0.4, γ = 0.1 (tuned per dataset/task).

### Short glossary (ResIntNet)

- **PRS (Perturbation-Response Scanning / Signal)**  
  Small virtual perturbations are applied at a site (e.g. active site), and the resulting response on the residue network is computed via a Laplacian/ENM model. Residues that show strong response or lie on high-flow edges are interpreted as part of allosteric communication pathways.

- **Memory conductance C**  
  An edge attribute that encodes how "important" a contact is for propagating signal. Inspired by adaptive flow networks, C is increased for edges that carry strong PRS flow and decreased for edges that rarely carry flow. The resulting pattern of high- vs low-C edges provides a history-dependent, non-linear refinement of standard ENM/PRS allostery maps.

- **GNN (Graph Neural Network)**  
  A neural network that operates directly on the residue-interaction graph, aggregating information from neighbors via message passing. Here, it complements the physics-based PRS/memory signals by learning patterns across proteins and mutation datasets.

### B.1 (Optional) Ingest external mutation datasets

Use adapters in `src/resintnet/ingest/adapters/` to normalize curated mutation sets (e.g., D3DistalMutation). These can be used to validate and calibrate the PRS/memory + GNN scores against experimentally known distal mutations or allosteric sites.

```bash
# Example: D3Distal CSV -> normalized parquet
python packages/resintnet/scripts/ingest_mutations.py \
  --adapter d3distal \
  --input data/mutations/D3DistalMutation.csv \
  --out data/mutations/normalized.parquet \
  --mapping configs/mappings/generic_example.yaml
```

Add new sources by making a small adapter module under `adapters/` and registering it.

### B.2 Rank residues (PRS + optional GNN)

```bash
python packages/resintnet/scripts/rank_mutations.py \
  --fasta data/fasta/10_subset.fa \
  --priors-dir data/templates/priors \
  --emb-dir data/emb/esm2_t12_plus_msa \
  --out-dir data/resintnet/scores \
  --alpha 0.5 \
  --beta 0.4 \
  --gamma 0.1 \
  --contact-threshold 8.0 \
  --topk-priors 400 \
  --save-graphs
```

This writes per-protein CSVs with:
- `residue_index`
- `score_prs` (normalized PRS centrality)
- `score_gnn` (if trained)
- `memory_focus` (edge/path-based memory signal)
- `score_blend` (final ensemble score)

**Training the GNN:** use `pipelines/train_resintnet.py` as a minimal example that samples nodes/graphs from your proteins, optionally supervised by curated mutation labels (if available).

---

## Stage C — SeqML (mutant efficacy / fitness modeling)

**Input:** hotspot residues (from ResIntNet) + wild-type sequences (+ experimental datasets if training).  
**Output:** fine-tuned model & predictions for small mutational neighborhoods (1–3 AA changes) around hotspots.

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
# Example: LoRA fine-tune a small T5 on curated fitness (if available)
python packages/seqml/scripts/train.py \
  --train data/seqml/train.csv \
  --val   data/seqml/val.csv \
  --out   data/seqml/t5_lora_run1

# Or run zero/few-shot inference (see package README)
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

---

## Configs

### Example `configs/pipeline.example.yaml`

```yaml
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
  alpha_blend: 0.5   # α in the blend
  beta_blend: 0.4    # β
  gamma_blend: 0.1   # γ
  save_graphs: true

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

- **Apple Silicon** works well with PyTorch MPS for ESM2 embedding and small models. Keep batch sizes small.
- **Remote MSA** avoids heavy local installs. If you need local MMseqs2, isolate it in a separate env/docker image.
- Use `pip install -e` per package to keep dependency islands small.
- Dockerfiles in `docker/` (coming) will pin OS/driver combos for reproducibility.

---

## Licenses & Data

- **Code** is under your repo's license (TBD). Third-party libraries keep their own licenses.
- **External datasets** (e.g., D3DistalMutation) may have usage restrictions. Confirm license terms before training or redistribution.
- If you add new dataset adapters, document the source and license in `packages/resintnet/src/resintnet/ingest/README.md`.

---

## What's next

- Add CI in `.github/workflows/` to lint & run a tiny smoke test per package.
- Publish minimal Docker images for each package.
- Optional: move persistent artifacts to cloud object storage and add a DW schema when you outgrow local caches.

**Happy hacking!**
