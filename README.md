
### protein-engineering — end‑to‑end sequence→structure→mutation ranking

This monorepo hosts an end‑to‑end, laptop‑friendly pipeline for protein engineering:
given **sequence(s)** and optional **experimental data**, we derive **structure‑aware contact/ distance priors** (ResContact), build a **residue‑interaction network** (ResIntNet), and **rank distal mutation candidates** with a blend of **GNN scoring** and **memory/PRS centrality**. Optionally, a SeqML package fine‑tunes sequence models to predict mutant efficacy.

> Works on MacBook Air M‑series (8 GB) with careful knobs (chunked ESM2, small GNNs, remote MSAs).

---

#### A. Repository layout (monorepo)

```
protein-engineering/
├── README.md
├── archive/
│   └── msa_utils/                 # legacy or experimental (kept for reference)
├── configs/                       # repo-level defaults (paths, bins, registries)
├── docker/                        # base images, compose
├── packages/
│   ├── rescontact/                # contact/distance priors (templates, ESM2, MSA)
│   │   ├── README.md
│   │   ├── notebooks/
│   │   ├── pyproject.toml
│   │   └── src/rescontact/
│   │       ├── data/              # PDB helpers
│   │       ├── features/          # ESM2, MSA, pair features
│   │       ├── models/            # (optional) contact nets
│   │       └── utils/             # metrics, train loops
│   ├── resintnet/                 # residue-interaction network + mutation ranking
│   │   ├── README.md
│   │   ├── notebook/
│   │   ├── pyproject.toml
│   │   └── src/resintnet/
│   └── seqml/                     # sequence-level modeling (mutant efficacy/fitness)
│       ├── README.md
│       ├── notebook/
│       ├── prot_api_flask/
│       └── pyproject.toml
└── pipelines/                     # thin orchestration (no heavy logic)
```

---

#### B. End‑to‑end flow

1) **ResContact** (packages/rescontact)
   - Input: FASTA (+ local PDB cache), remote MSA via MMseqs API.
   - Output:
     - `emb/`: per‑residue ESM2 embeddings (`L × C`)
     - `msas/` + `msa_features/`: A3M + conservation/MI/APC summaries
     - `priors/`: `(L × L × B)` distance‑bin priors from templates (PDB/AFDB); `bins.npy`, `mask.npy`, and `meta.json` per query.

2) **ResIntNet** (packages/resintnet)
   - Build residue‑interaction graph from priors/structure:
     - Nodes = residues; node features = ESM2 + MSA summaries (+ optional physchem).
     - Edges = contacts from CA distance threshold or top‑K from priors; edge attrs = binned distance, 1/r, template agreement, etc.
   - Score residues with:
     - **GNN** (GraphSAGE/GAT) → per‑residue logits/scores.
     - **PRS/Memory** → perturbation response centrality per residue.
   - **Blend** = `final_score = α · sigmoid(gϕ) + (1−α) · norm(PRS)`.
   - Output: ranked residues (distal candidates), CSV per protein.

3) **SeqML** (packages/seqml)
   - Fine‑tune sequence models (e.g., T5/ESM/LLM) to predict mutant fitness/efficacy using curated experimental datasets; take top‑K residues from ResIntNet and enumerate small mutational neighborhoods.

---

#### C. Remote MSA (MMseqs) + ESM2 + Priors (quick recap)

```bash
1) FASTA → ESM2 embeddings (tiny model OK on M‑series)
python packages/rescontact/scripts/embed_esm2.py \
  --fasta data/fasta/10_subset.fa \
  --out-dir data/emb/esm2_t12 \
  --model esm2_t12_35M_UR50D

2) Remote MSA (rate-limited; no local install required)
python packages/rescontact/scripts/run_msa_batch.py \
  --fasta data/fasta/10_subset.fa \
  --msa-out-dir data/msas \
  --server-url https://a3m.mmseqs.com \
  --db uniref --qps 0.15

3) MSA → compact features (depth, PSSM/PSFM, MI/APC summaries)
python packages/rescontact/scripts/build_msa_features.py \
  --msa-dir data/msas \
  --esm-emb-dir data/emb/esm2_t12 \
  --out-dir data/msa_features \
  --float16

4) Template priors (PDB/AFDB) → (L×L×B) distances
export RESCONTACT_TEMPLATE_DIR=data/templates
python packages/rescontact/scripts/build_template_priors.py \
  --hits data/templates/mmseqs_hits.json \
  --pdb-root data/pdb/train data/pdb/test \
  --out-dir "$RESCONTACT_TEMPLATE_DIR/priors" \
  --structure-source "pdb,afdb" \
  --max-hits-per-query 8 \
  --max-downloads-per-run 50
```

Outputs you’ll reuse in ResIntNet:
- `data/emb/esm2_t12/{ID}.esm2.npy` (L×C)
- `data/msa_features/{ID}.npz` with `{X (L×F), depth, meta}`
- `data/templates/priors/{ID}.npz` with `{priors (L×L×B), bins, mask, meta}`

---

#### D. **Supervised training with distal‑mutation datasets** (NEW)

You **can** supervise the GNN ranking using curated distal‑mutation sets (e.g., literature compilations). The idea:

- Convert *mutation‑level* evidence into **residue‑level labels**.
- Train a **per‑residue classifier** (or **pairwise ranker**) on graphs built from priors/structure and node features (ESM2, MSA, etc.).
- Blend GNN scores with **PRS/Memory centrality** at inference.

> ⚠️ Always check the dataset’s license/terms. Keep raw third‑party data out of the repo; add a small downloader/normalizer instead.

##### D.1 Normalized CSV schema

Create `data/supervision/distal_mutations.csv`:

| column            | type    | notes                                                                 |
|-------------------|---------|-----------------------------------------------------------------------|
| protein_id        | str     | e.g., `1ABC_A` (PDB ID + chain) or your internal query ID            |
| chain_id          | str     | single letter; optional if encoded in `protein_id`                    |
| residue_index_seq | int     | 1‑based **sequence** index (align to FASTA/ESM2)                      |
| label             | int     | 1 = distal hotspot residue, 0 = non‑hotspot                           |
| effect_size       | float   | optional (Δactivity, Δstability, …)                                   |
| evidence          | str     | optional citation / source                                            |
| split             | str     | `train` / `val` / `test` (or generate via script)                     |

> If your source data uses PDB residue numbering, map to sequence indices via your existing mapper (`rescontact.data.pdb_utils`) to align with ESM2 embeddings.

**Residue label aggregation:** if any mutation at a residue is curated as distal/beneficial (per your criterion), mark the residue `label=1`. Otherwise, sample negatives from residues not known positive (optionally exclude active‑site residues or zones too close to the binding site to focus on distal candidates).

##### D.2 Graph construction (per protein)

- **Nodes (L):** residues 1..L
  - `x_node = concat([ESM2_i, MSA_i, optional physchem_i])`
- **Edges:** build from structure/priors
  - Undirected edges for `CA_dist < 8Å` **or** top‑K neighbors from priors (e.g., `K=16` per residue).
  - Edge attrs: `[1/dist, one_hot(dist_bin), template_agreement, …]`
- **Labels:** `y_i ∈ {0,1}` from `distal_mutations.csv` (missing → mask as unlabeled during training).

##### D.3 Objective choices

- **Binary classification** (default): per‑residue BCEWithLogitsLoss with **class weights** (positives are rare).
- **Pairwise ranking**: sample (pos, neg) pairs within a protein; BPR/hinge loss to push positive residues above negatives.
- **Multitask**: classification + auxiliary regression on `effect_size` (if available).

### D.4 Train/Val/Test split

- **By protein**, not by residue, to avoid leakage. A simple split file:
  - `data/supervision/splits.json`: `{ "train": [...], "val": [...], "test": [...] }`
- Or assign in the CSV `split` column.

##### D.5 Minimal training CLI

Prepare tensors/graphs:
```bash
# Build per-protein graphs + labels (reads ESM2/MSA/priors; writes .pt files)
python packages/resintnet/scripts/prepare_graphs.py \
  --esm-dir data/emb/esm2_t12 \
  --msa-dir data/msa_features \
  --priors-dir data/templates/priors \
  --labels-csv data/supervision/distal_mutations.csv \
  --out-dir data/graphs \
  --dist-threshold 8.0 \
  --topk-from-priors 16 \
  --add-edge-bins
```

Train a small GraphSAGE:
```bash
python packages/resintnet/scripts/train_gnn.py \
  --graphs-dir data/graphs \
  --model graphsage \
  --hidden 256 --layers 3 --dropout 0.3 \
  --loss bce \
  --class-weight-pos 8.0 \
  --epochs 50 --batch-size 1 \
  --device mps   # or cpu
```

Evaluate + export ranked residues:
```bash
python packages/resintnet/scripts/rank_mutations.py \
  --graphs-dir data/graphs \
  --ckpt runs/gnn/best.ckpt \
  --prs-alpha 0.4 \
  --out-dir outputs/ranks
```

Outputs: `outputs/ranks/{protein_id}.csv` with `residue_index_seq, gnn_score, prs, final_score` (descending).

##### D.6 Blending with PRS/Memory

- Compute **PRS centrality** per residue (packages/resintnet has a `prs.py` util).
- Normalize to `[0,1]` across a protein.
- Blend: `final = α·sigmoid(gnn_logits) + (1−α)·prs_norm`, with `α∈[0.3,0.7]` (tune on `val`).

### D.7 Metrics

- **Per‑protein**: AUROC, AUPRC, Hit@K (K=5,10,20), Enrichment@K.
- **Global**: macro/micro AUPRC across proteins.
- **Ablations**: w/ vs w/o PRS; w/ vs w/o MSA; priors vs geometry edges.

---

#### E. Notes on resources & licensing

- **Third‑party datasets**: keep outside the repo and provide a download/prepare script; verify license allows model training.
- **Apple Silicon**: prefer smaller ESM2 (`t12_35M`), float16 storage, shallow GNNs; avoid giant minibatches.
- **Reproducibility**: set seeds; use cross‑protein splits; log config in `runs/`.
