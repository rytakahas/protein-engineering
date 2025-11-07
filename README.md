# protein-engineering

> End‑to‑end toolkit for **protein engineering** and **small‑molecule/enzyme design**.  
> Input: sequences + experimental datasets.  
> Output: **hotspots (distal mutation candidates)** and ranked variants, using **structure/contact priors**, **residue‑interaction networks**, and **sequence‑level ML**.

This repository is a **monorepo** with three main packages plus thin orchestration pipelines:

```
├── README.md
├── archive
│   └── msa_utils/
│       ├── Dockerfile
│       ├── README.md
│       └── run_deepmsa.sh
├── configs/
├── docker/
├── packages/
│   ├── rescontact/              # Template priors (PDB/AFDB), ESM2 embeds, MSA features → contact/distance priors
│   │   ├── README.md
│   │   ├── notebooks/
│   │   │   ├── res_contact_workflow.ipynb
│   │   │   ├── res_contact_workflow_opt.ipynb
│   │   │   ├── res_contact_workflow_opt2.ipynb
│   │   │   └── res_contact_workflow_opt_msa.ipynb
│   │   ├── pyproject.toml
│   │   └── src/rescontact/
│   │       ├── __init__.py
│   │       ├── data/pdb_utils.py
│   │       ├── features/
│   │       │   ├── embedding.py
│   │       │   ├── msa.py
│   │       │   └── pair_features.py
│   │       ├── models/contact_net.py
│   │       └── utils/{metrics.py, train.py}
│   ├── resintnet/               # Residue interaction networks + memory‑theory mutation ranking
│   │   ├── README.md
│   │   ├── notebook/
│   │   │   ├── README.md
│   │   │   ├── prot_rin_gnn.ipynb
│   │   │   └── prot_rin_mem.ipynb
│   │   └── pyproject.toml
│   └── seqml/                   # Sequence‑level modeling: fitness/efficacy for proposed mutations
│       ├── README.md
│       ├── notebook/
│       │   ├── PT5_xl_ACT.ipynb
│       │   ├── PT5_xl_GB1.ipynb
│       │   ├── README.md
│       │   ├── msa.ipynb
│       │   ├── protBert_ACT.ipynb
│       │   └── unirep_ACT.ipynb
│       ├── prot_api_flask/
│       │   ├── README.md
│       │   ├── mutation_predictor.py
│       │   ├── prot_api_flask.py
│       │   └── sample_input.json
│       └── pyproject.toml
└── pipelines/                   # thin glue/DAGs, no heavy logic
```

> **Note on naming**
> - Keep **`rescontact`** (don’t rename to “res‑res‑contact”).  
> - Standardize on **`resintnet`** for residue‑interaction‑network code.  
> - Remove redundant wrapper folders (e.g., `Res_Int_Net/`, `Seq_MLs/`) over time by migrating code into `src/<package>/`.

---

## 1) High‑level workflow

```text
Sequences (+optional experimental labels)
   │
   ├─► packages/rescontact
   │     1) Build structure priors from templates (PDB/AFDB) → distance bins
   │     2) Get ESM2 embeddings (single‑seq) → L × C
   │     3) Build MSA features (via mmseqs remote A3M) → profiles, depth
   │     4) Concatenate features, train ContactNet → P(contact/dist bins)
   │
   ├─► packages/resintnet
   │     5) Build residue‑interaction graph (from contacts/structure)
   │     6) Rank mutation hotspots using memory/graph signals (GNN/PRS/etc.)
   │
   └─► packages/seqml
         7) Score candidate mutations (LMs, regressors, calibration with assays)
         8) Feedback loop with new experimental results
```

### Minimal ASCII of feature flow
```
Templates (PDB/AFDB) ─┐
                      ├─► Priors (L×L×B) → ContactNet
ESM2 embeddings  ─────┤
MSA features     ─────┘
          ContactNet → P(contact/dist) → RIN → mutation ranking → SeqML scoring
```

---

## 2) Getting started

### Dev environment
- Python **3.11**
- PyTorch with **MPS** (Apple Silicon) or CUDA
- Optional: `mmseqs` **remote API** (no local install needed)

```bash
# clone
git clone https://github.com/rtakahas/protein-engineering.git
cd protein-engineering

# optional: install shared tooling
pip install -e packages/rescontact
pip install -e packages/resintnet
pip install -e packages/seqml
```

> **Sparse‑checkout (clone only a subfolder)**  
> If you only want `packages/rescontact`:
> ```bash
> git clone --no-checkout https://github.com/rtakahas/protein-engineering.git
> cd protein-engineering
> git sparse-checkout init --cone
> git sparse-checkout set packages/rescontact
> git checkout
> ```

---

## 3) End‑to‑end (contact priors → hotspots → sequence scoring)

Below is a lightweight **reference** pipeline using scripts in `rescontact`. Adjust paths as needed.

### 3.1 Prepare FASTA from PDB (optional)
```bash
PYTHONPATH=packages/rescontact/src \
python packages/rescontact/scripts/pdb_to_fasta.py \
  --pdb-root data/pdb/train data/pdb/test \
  --out data/fasta/_subset.fa
```

### 3.2 Retrieve homologs & template hits (remote mmseqs, A3M endpoint)
```bash
PYTHONPATH=packages/rescontact/src \
python packages/rescontact/scripts/retrieve_homologs.py \
  --fasta data/fasta/_subset.fa \
  --server-url https://a3m.mmseqs.com \
  --db uniref \
  --max-hits 16 \
  --want-templates \
  --out data/templates/mmseqs_hits.json
```

### 3.3 Build template priors (distance bins, masks)
```bash
export RESCONTACT_TEMPLATE_DIR=data/templates

PYTHONPATH=packages/rescontact/src \
python packages/rescontact/scripts/build_template_priors.py \
  --hits data/templates/mmseqs_hits.json \
  --pdb-root data/pdb/train data/pdb/test \
  --out-dir "$RESCONTACT_TEMPLATE_DIR/priors" \
  --structure-source "pdb,afdb" \
  --max-hits-per-query 8 \
  --max-downloads-per-run 50
```

**Outputs (per query)**  
- `priors`: `(L, L, B)` distance‑bin probabilities (normalized)  
- `bins`: distance bin edges used during training  
- `mask`: `(L, L)` valid prior region  
- `meta`: JSON string (query_id, L, templates_used, etc.)  
- Downloaded PDB/AFDB cached to avoid re‑fetching

### 3.4 Get MSAs (mmseqs remote) and turn into features
```bash
# Run remote MSA (A3M + tgz per sequence)
PYTHONPATH=packages/rescontact/src \
python packages/rescontact/scripts/run_msa_batch.py \
  --fasta data/fasta/_subset.fa \
  --msa-out-dir data/msas \
  --server-url https://a3m.mmseqs.com \
  --db uniref \
  --qps 0.15

# Convert A3M → per‑residue features (PSSM/PSFM, MI/APC summaries, depth)
PYTHONPATH=packages/rescontact/src \
python packages/rescontact/scripts/build_msa_features.py \
  --msa-dir data/msas \
  --esm-emb-dir data/emb/esm2_t12 \
  --out-dir data/msa_features \
  --float16
```

### 3.5 Embed with ESM2 (single‑sequence)
```bash
PYTHONPATH=packages/rescontact/src \
python packages/rescontact/scripts/embed_esm2.py \
  --fasta data/fasta/_subset.fa \
  --out-dir data/emb/esm2_t12 \
  --model esm2_t12_35M_UR50D
```

### 3.6 Concatenate ESM2 + MSA features
```bash
PYTHONPATH=packages/rescontact/src \
python packages/rescontact/scripts/concat_esm2_msa.py \
  --esm-dir data/emb/esm2_t12 \
  --msa-dir data/msa_features \
  --out-dir data/emb/esm2_t12_plus_msa \
  --mode pad \
  --include-depth \
  --float16 \
  --verbose
```

### 3.7 Train ContactNet (distance/contact prediction)
```bash
PYTHONPATH=packages/rescontact/src \
python packages/rescontact/scripts/train_contactnet.py \
  --emb-dir data/emb/esm2_t12_plus_msa \
  --priors-dir data/templates/priors \
  --bins-json data/templates/priors/bins.json \
  --out-dir outputs/rescontact_model \
  --epochs 50 --lr 3e-4 --batch-size 2 --device mps
```

### 3.8 Build Residue Interaction Network (RIN) + rank hotspots
```bash
PYTHONPATH=packages/resintnet/src \
python packages/resintnet/scripts/build_graph.py \
  --contacts outputs/rescontact_model/preds \
  --out data/rin/graphs

PYTHONPATH=packages/resintnet/src \
python packages/resintnet/scripts/rank_mutations.py \
  --graphs data/rin/graphs \
  --method mem  \
  --out data/rin/ranked_mutations.csv
```

### 3.9 Score variants with SeqML
```bash
PYTHONPATH=packages/seqml/src \
python packages/seqml/scripts/score_variants.py \
  --wt-fasta data/fasta/wt.fa \
  --candidates data/rin/ranked_mutations.csv \
  --train-table data/assays/train.csv \
  --out data/seqml/scores.csv
```

> **Feedback loop:** append new assay results to `data/assays/` and re‑train `seqml` for better calibration.

---

## 4) Data conventions

```
data/
├── fasta/                 # input sequences
├── pdb/                   # local PDB/AFDB mirrors (optional)
├── templates/
│   ├── mmseqs_hits.json   # remote hits (with/without template alignment metadata)
│   └── priors/            # *.npz with priors, mask, meta, bins.json
├── msas/                  # *.a3m and *.tgz
├── emb/
│   ├── esm2_t12/          # *.esm2.npy (L×C)
│   └── esm2_t12_plus_msa/ # concatenated features
└── assays/                # experimental labels / fitness data
```

- **Remote MSA (mmseqs)**: No local `colabfold` needed. We call the official A3M server, respecting rate limits.  
- **ESM2**: Single‑sequence embeddings only (no multiple sequence transformer needed).  
- **Template priors**: Use both PDB and AFDB, falling back gracefully.

---

## 5) Containerization

Base images are under `docker/`. Each package can have its own `Dockerfile` or reuse a shared base.

Example `docker-compose.yaml` sketch:
```yaml
services:
  rescontact:
    build:
      context: .
      dockerfile: docker/base.Dockerfile
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: bash -lc "pip install -e packages/rescontact && pytest packages/rescontact"
  resintnet:
    build:
      context: .
      dockerfile: docker/base.Dockerfile
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: bash -lc "pip install -e packages/resintnet && pytest packages/resintnet"
  seqml:
    build:
      context: .
      dockerfile: docker/base.Dockerfile
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: bash -lc "pip install -e packages/seqml && pytest packages/seqml"
```

---

## 6) Makefile helpers (suggested)
```make
.PHONY: setup fmt lint test

setup:
\tpip install -e packages/rescontact -e packages/resintnet -e packages/seqml
\tpip install -r requirements-dev.txt

fmt:
\truff format
\truff check --fix

lint:
\truff check
\tmypy packages

test:
\tpytest -q
```

---

## 7) Roadmap

- [ ] Migrate remaining code from legacy wrappers into `src/<pkg>/`
- [ ] Add `resintnet/src` scripts (`build_graph.py`, `rank_mutations.py`)
- [ ] Publish example configs in `configs/` (binning, thresholds, server QPS)
- [ ] Add CI with smoke notebooks and unit tests
- [ ] Optional local MMseqs & AFDB cache containers (for offline runs)
- [ ] Integration pipeline under `pipelines/`:
  - `train_rescontact.py`, `train_seqml.py`, `e2e_propose_mutations.py`

---

## FAQ

**Do I need ColabFold installed?**  
No. For *remote MSAs*, we call `https://a3m.mmseqs.com` directly. Install ColabFold only if you want local searches.

**Is `rescontact` the same as structure prediction?**  
It focuses on **contact/distance priors** (template + ESM2 + MSA) for training a lightweight contact net—useful for graph building and downstream mutation ranking.

**Why ESM2 single‑sequence?**  
It’s fast/light, works on 8‑GB Mac with MPS, and combines well with MSA summaries + template priors.

---

## Citation & License

- See individual package `README.md` for academic references (ESM2, MMseqs, AFDB, RIN/PRS, etc.).  
- License: MIT (unless specified otherwise in a package).
