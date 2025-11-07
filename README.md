# protein-engineering — E2E mutation hotspot discovery (sequence ↔ structure)

This monorepo targets an **end‑to‑end pipeline** for protein & enzyme engineering:
given **sequence(s)** + **experimental datasets**, we predict **distal mutation hotspots**
guided by **structure/contact information** and **sequence models**, then rank candidates for wet‑lab validation.

> High‑level flow  
> **Sequences → (Templates + ESM2 + MSA) → Contact/Distance Priors → Residue‑Interaction Graph → PRS/Memory + GNN ranking → Candidate mutations → Sequence‑level ML (fine‑tuned LLMs) → Predicted efficacy**

---

## Repository layout (monorepo)

```
protein-engineering/
├── README.md
├── archive/
│   └── msa_utils/                 # legacy scripts (kept for reference)
├── configs/                       # repo-level defaults (paths, binning, registry)
├── docker/                        # base images & compose
├── packages/
│   ├── rescontact/                # contact/distance priors (Templates, ESM2, MSA)
│   │   ├── README.md
│   │   ├── notebooks/
│   │   ├── pyproject.toml
│   │   └── src/rescontact/
│   │       ├── data/pdb_utils.py
│   │       ├── features/          # embedding.py, msa.py, pair_features.py
│   │       ├── models/contact_net.py
│   │       └── utils/             # metrics.py, train.py
│   ├── resintnet/                 # residue-interaction network + hotspot ranking
│   │   ├── README.md
│   │   ├── notebook/
│   │   └── pyproject.toml
│   └── seqml/                     # sequence-level modeling (mut efficacy/fitness)
│       ├── README.md
│       ├── notebook/
│       ├── prot_api_flask/
│       └── pyproject.toml
└── pipelines/                     # thin glue/DAGs; no heavy logic
```

**Why this shape?**
- Each `packages/*` subdir is an **installable Python package** (with its own tests, configs, Dockerfile).
- `pipelines/` holds light orchestration (CLI or DAG) that **composes packages** without duplicating logic.
- `archive/` contains legacy/experimental code kept out of the critical path.

---

## Quickstart (Apple Silicon friendly)

> Recommended: Python **3.11**, PyTorch **2.5+**, CUDA not required on Mac; uses MPS fallback.

```bash
# 1) Clone
git clone https://github.com/rtakahas/protein-engineering.git
cd protein-engineering

# 2) Create env
conda create -y -n pe311 python=3.11
conda activate pe311

# 3) Install base (editables)
pip install -e packages/rescontact -e packages/resintnet -e packages/seqml

# (Optional) Apple Metal fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Data folders (local, gitignored)
```
data/
  fasta/               # input sequences (.fa)
  templates/           # mmseqs hits + priors caches
  emb/                 # ESM2 embeddings (per-L npy)
  msas/                # A3M files + features
```

---

## Stage A — Contacts/Distance Priors (packages/rescontact)

This stage builds **structure-aware priors** per query sequence by fusing:
- **Template alignments** (from PDB/AFDB hits) → distance-bin histograms
- **ESM2 residue embeddings** (per-position)
- **MSA‑derived features** (e.g., PSSM/PSFM; optional)

### A1) Prepare FASTA
If your PDB chains are already parsed, you likely have `data/fasta/_subset.fa`. Otherwise:
```bash
# Example (pseudo): convert PDB/CIF to FASTA
python -m rescontact.data.pdb_utils \
  --pdb-root data/pdb/train data/pdb/test \
  --out data/fasta/_subset.fa
```

### A2) Remote homology & templates via MMseqs API
**No ColabFold install required**; we use the public MMseqs/A3M API.

```bash
export RESCONTACT_TEMPLATE_DIR=data/templates

python scripts/retrieve_homologs.py \
  --fasta data/fasta/10_subset.fa \
  --server-url https://a3m.mmseqs.com \
  --db uniref \
  --max-hits 16 \
  --want-templates \
  --qps 0.15 --inter-job-sleep 2 --max-retries 8 --timeout 1800 \
  --flush-every 1 \
  --out data/templates/mmseqs_hits.json
```

> Tip: If you hit rate‑limits (HTTP 429), reduce `--qps` and keep `--resume` enabled.

### A3) Build **template priors** (distance bins)
This aligns selected templates to the query and outputs per‑pair distance probabilities.

```bash
python scripts/build_template_priors.py \
  --hits data/templates/mmseqs_hits.json \
  --pdb-root data/pdb/train data/pdb/test \
  --out-dir "$RESCONTACT_TEMPLATE_DIR/priors" \
  --structure-source "pdb,afdb" \
  --max-hits-per-query 8 \
  --max-downloads-per-run 50
# -> writes: data/templates/priors/<QUERY>.npz
#    keys: priors (L,L,B), bins (B+1), mask (L,L), meta (json)
```

### A4) **ESM2 embeddings**
```bash
python scripts/embed_esm2.py \
  --fasta data/fasta/10_subset.fa \
  --out-dir data/emb/esm2_t12 \
  --model esm2_t12_35M_UR50D
# -> <QUERY>.esm2.npy  shape=(L, C)
```

### A5) **MSA** (A3M) and features (optional but recommended)
```bash
# Download A3M
python scripts/run_msa_batch.py \
  --fasta data/fasta/10_subset.fa \
  --msa-out-dir data/msas \
  --server-url https://a3m.mmseqs.com \
  --db uniref \
  --qps 0.15

# Build per‑position MSA features (PSSM/PSFM; depth stats)
python scripts/build_msa_features.py \
  --msa-dir data/msas \
  --esm-emb-dir data/emb/esm2_t12 \
  --out-dir data/msa_features \
  --float16

# (Optional) Concatenate ESM2 + MSA to a single feature tensor
python scripts/concat_esm2_msa.py \
  --esm-dir data/emb/esm2_t12 \
  --msa-dir data/msa_features \
  --out-dir data/emb/esm2_t12_plus_msa \
  --mode pad \
  --include-depth \
  --float16
```

### A6) Train contact/distance model (planned CLI)
Planned `pipelines/train_rescontact.py` (thin wrapper) to train with:
- Inputs: `emb_dir` (ESM2 or ESM2+MSA), `priors_dir` (template npz), bin edges
- Loss: cross‑entropy on distance bins + aux contact loss
- Splits: protein‑level to avoid leakage

> For now, see `packages/rescontact/src/rescontact/utils/train.py` to wire your datasets.

---

## Stage B — Residue‑Interaction Network (packages/resintnet)

Build a **graph over residues** using predicted/contact priors &/or a structure model, then rank **distal mutation hotspots** with **memory/PRS** and a **GNN**.

**Planned workflow:**
1. **Graph build**: nodes = residues, edges weighted by contact prob / distances.  
   ```bash
   python -m resintnet.scripts.build_graph \
     --contacts data/templates/priors \
     --out data/graphs
   ```
2. **PRS/Memory**: compute perturbation response & centrality per residue.
3. **GNN ranking**: train GraphSAGE/GAT with node/edge features (ESM2, priors).  
   Output: ranked residues (distal candidates).

### External mutation datasets (for supervision)
If you use curated distal‑mutation datasets (e.g., literature/third‑party), ensure the **license allows ML training**. Prepare a normalized CSV:
```
protein_id, pdb_id, chain, position, wt_aa, mut_aa, label_distal, assay, effect_value
```

> Split **by protein** (not by mutation) to avoid optimistic leakage. Evaluate Top‑k precision/recall, AUPRC, and NDCG@k.

---

## Stage C — Sequence‑level ML (packages/seqml)

Fine‑tune sequence models (e.g., ESM2/ProtT5) to score **WT→mutant** efficacy (stability, activity, binding). Support **LoRA/QLoRA** for resource‑friendly finetuning.

**Inputs:**
- Mutant sequences or edits (e.g., `A123C`) and optional structural features (from Stage B)
- Experimental labels (ΔΔG, activity, etc.)

**Outputs:**
- Regression/class scores per mutation
- Re‑ranking of hotspot candidates

---

## Putting it together: one‑shot pipeline

```
pipelines/e2e_propose_mutations.py
  1) A2–A5: build priors + embeddings + MSA
  2) B1–B3: construct RIN & rank distal hotspots
  3) C*: score WT→mutants; output top‑N list + rationale
```

---

## Docker & Reproducibility

Each package can define its own `Dockerfile` (pinning Python & deps). Suggested base:
```
docker/
  base.Dockerfile   # uv/pip-tools, torch cpu/mps, biopython>=1.83
  docker-compose.yml
```

### Known version pitfalls
- Avoid installing `colabfold` unless needed; it pins **old numpy/biopython**.
- Prefer **biopython ≥ 1.83** (some code expects it).
- On macOS MPS: `TORCH_MPS_HIGH_WATERMARK_RATIO=0.0` may help long runs.

---

## Datasets & Licensing

You are responsible for ensuring any external datasets you use are **permitted for ML training** in your context.  
Keep raw data paths out of git; place loaders in `packages/*/dataio` with clear schemas.

---

## Roadmap (short)

- [ ] `pipelines/train_rescontact.py` CLI
- [ ] `resintnet` PRS + GNN scripts
- [ ] `seqml` LoRA fine‑tuning CLI + evaluation harness
- [ ] End‑to‑end Airflow/Prefect DAG & Makefile targets

---

## Troubleshooting

- **MMseqs API 429/307**: lower `--qps`; use `--resume`; ensure `--want-templates` only when needed.
- **IndexError in MSA features**: ensure query FASTA and A3M columns align (script now guards with column mapping & insertions).
- **Version conflicts**: create a clean env; pin torch, numpy, biopython; avoid mixing TF/ColabFold in the same env.

---

## Citation & Acknowledgments

This repo composes public building blocks (ESM2, MMseqs, PDB/AFDB resources). Please cite upstream projects where appropriate.  
Contributions welcome — follow `CONTRIBUTING.md` (to be added).
