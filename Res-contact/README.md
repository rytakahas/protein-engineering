# Res-Contact — ESM2 + (optional) MSA + (optional) Template Priors

> **One‑liner:** Freeze ESM2 for per‑residue embeddings, optionally fuse **MSA 1‑D features** and **homology‑derived structural priors** (from MMseqs2→PDB/AFDB) to predict Cα–Cα ≤ 8 Å contacts. Everything is **cache‑first** and designed to run on an **8‑GB MacBook Air (M3)**.

---

## What changed in this update (MSA + PDB templates + re‑aligned ground truth)

- Added a **remote‑only MMseqs2** flow that returns **PDB template IDs + residue alignments** (no local UniRef DB needed).
- New, robust **`retrieve_homologs.py`** (with `--want-templates`, QPS limiting, retries, and resume) so we can get **PDB/AFDB template candidates** with `qstart/qend/pident/...`.
- **`build_template_priors.py`** now consumes those alignments, **downloads** the matched structures (PDB/AFDB), **maps residues** to your query sequence, and builds **distance‑bin priors** per query into **`*.npz`** files.
- Dedicated MSA path: **`run_msa_batch.py` → `build_msa_features.py` → `concat_esm2_msa.py`**, producing **1‑D MSA statistics** (AA frequencies + entropy) aligned to each query.
- README now documents the full **end‑to‑end**: **PDB → FASTA → MMseqs2 templates → priors → (optional) MSA → ESM2 embeddings → concat → train/eval**.
- Ground‑truth labels are **recomputed** from your **PDB/mmCIF** (Cα–Cα ≤ 8 Å) at load time; template priors only act as **weak hints** during training/inference.

---

## Hardware & env tips (macOS / MPS)

```bash
conda create -n rescontact python=3.11 -y
conda activate rescontact
pip install -r requirements.txt

# Apple GPU knobs
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1   # if you hit unsupported ops
```

> **Avoid package pin‑wars:** You do **not** need to install ColabFold locally for this flow. We call the **hosted MMseqs2** API. If you *do* need ColabFold later, use a **separate env** to avoid conflicts (e.g., `biopython`, `typing_extensions`, `numpy`).

---

## Data & cache layout

```
data/
  pdb/
    train/                          # your PDB/mmCIF for training
    test/                           # your held‑out structures
  fasta/
    _subset.fa                      # sequences extracted from PDB
  templates/
    mmseqs_hits.json                # homolog hits (PDB + UniRef)
.cache/rescontact/
  templates/
    priors/                         # per‑query .npz with priors
embeddings/
  esm2_t12/                         # per‑query ESM2 .npy
  esm2_t12_plus_msa/                # ESM2 concatenated with MSA features
msas/
  raw_a3m/                          # A3M files per query
  features/                         # MSA 1‑D stats (npz)
artifacts/
  ...                               # trained models, logs, reports
```

---

## End‑to‑end pipeline

### 1) Extract sequences from PDB (SEQRES‑first, ATOM fallback)
```bash
python scripts/pdb_to_fasta.py \
  --pdb-root data/pdb/train \
  --out data/fasta/_subset.fa \
  --min-len 30 \
  --limit-files 200 \
  --verbose
```
- Works with **symlinks** or real files.
- Logs show detected chain counts/lengths; duplicates are deduped.

### 2) Retrieve homologs **with template IDs + alignments** (remote MMseqs2)
```bash
python scripts/retrieve_homologs.py \
  --fasta data/fasta/_subset.fa \
  --server-url https://a3m.mmseqs.com \
  --db uniref \
  --max-hits 16 \
  --want-templates \
  --qps 0.15 --inter-job-sleep 2 --max-retries 12 --timeout 1800 \
  --resume --flush-every 1 \
  --out data/templates/mmseqs_hits.json
```
- The output now includes **PDB subjects** (e.g., `1p2l_A`) with **`pident`, `alnlen`, `qstart`, `qend`, `evalue`, `bits`**.
- **Rate‑limit‑friendly**: `--qps`, `--inter-job-sleep`, `--max-retries`, and `--resume` keep long runs stable.
- If the server returns HTTP **307** or **429**, the client transparently retries with backoff.

### 3) Build **template priors** (downloads PDB/AFDB, maps, bins distances)
```bash
export RESCONTACT_TEMPLATE_DIR=.cache/rescontact/templates

python scripts/build_template_priors.py \
  --hits data/templates/mmseqs_hits.json \
  --pdb-root data/pdb/train data/pdb/test \
  --out-dir "$RESCONTACT_TEMPLATE_DIR/priors" \
  --structure-source "pdb,afdb" \
  --max-hits-per-query 8 \
  --max-downloads-per-run 50 \
  --allow-naive-mapping    # drop this when server returns per‑residue alignments
```
This writes **`<QUERY>.npz`** containing:
- `priors`: **(L, L, B)** distance‑bin probabilities aggregated across templates
- `bins`: the **bin edges** used to build histograms
- `mask`: **(L, L)** boolean; True where priors exist
- `meta`: JSON (query_id, L, templates_used, sources)

> Structures are cached; re‑runs don’t redownload. Self‑hits are ignored during fusion.

### 4) (Optional) Generate **MSA** and turn it into **1‑D features**
1) MMseqs2 A3M per query:
```bash
python scripts/run_msa_batch.py \
  --fasta data/fasta/_subset.fa \
  --msa-out-dir msas/raw_a3m \
  --server-url https://a3m.mmseqs.com \
  --db uniref \
  --qps 0.15
```
2) Convert A3M → aligned 1‑D features (AA frequencies + entropy):
```bash
python scripts/build_msa_features.py \
  --msa-dir msas/raw_a3m \
  --esm-emb-dir embeddings/esm2_t12 \
  --out-dir msas/features \
  --float16
```
Outputs `*.msa.npz` with:
- `msa1d`: **(L, 21)** float array (20 AA freqs + entropy)
- `depth`: total effective sequences used (for optional normalization)
- `meta`: alignment stats

### 5) ESM2 embeddings (frozen backbone)
```bash
python scripts/embed_esm2.py \
  --fasta data/fasta/_subset.fa \
  --out-dir embeddings/esm2_t12 \
  --model esm2_t12_35M_UR50D
```
Writes `*.esm2.npy` with shape **(L, C)** (e.g., **C=480** for `t12_35M`).

### 6) Concatenate **ESM2 + MSA** (or skip if ESM2‑only)
```bash
python scripts/concat_esm2_msa.py \
  --esm-dir embeddings/esm2_t12 \
  --msa-dir msas/features \
  --out-dir embeddings/esm2_t12_plus_msa \
  --float16 \
  --mode pad \
  --include-depth \
  --verbose
```
- Result per query: **(L, C + 21 [+1 if depth])**.  
- If you skip MSA, just point the trainer to `embeddings/esm2_t12`.

### 7) Train (ESM2 only, +MSA, and/or +template priors)
```bash
python scripts/train.py \
  --config configs/rescontact.yaml \
  --save-dir artifacts/esm2_t12_templates_msa \
  --epochs 20 --batch-size 1
```
Key config knobs (see `configs/rescontact.yaml`):
```yaml
features:
  use_templates: true
  use_msa: true
templates:
  fuse_mode: logit_blend      # or "feature_concat"
  blend_alpha: 0.3            # prior-to-logit mixing weight
data:
  pdb_root: data/pdb          # where labels are recomputed (Cα ≤ 8 Å)
  max_len_per_chain: 600
train:
  pos_weight: 10.0            # optional class balance
```

### 8) Evaluate & visualize
```bash
python scripts/eval.py \
  --config configs/rescontact.yaml \
  --ckpt artifacts/esm2_t12_templates_msa/model_best.pt \
  --split test
```
Outputs PR‑AUC, ROC‑AUC, F1 (with configurable threshold), plus optional **P@L** (disabled by default; prefer long‑range P@k).

---

## Ground truth & leakage notes

- Labels are recomputed from **your PDB/mmCIF** at load time: **Cα–Cα ≤ 8 Å** = contact; diagonal & |i−j|<*min_sep* can be masked (see config).
- Template priors are **hints**, not labels. For fair evaluation, ensure your **train/val/test split** doesn’t leak the **same target structure** into template sources for that split. The code excludes **self‑hits** by default during fusion; you can also raise `min_ident` if desired.

---

## Troubleshooting (quick)

- **HTTP 429 / 307 from MMseqs2** → the client already retries with backoff; lower `--qps`, add `--inter-job-sleep`, and keep `--resume` on.
- **Biopython / typing_extensions conflicts** → don’t mix ColabFold & training deps in one env.
- **MPS oddities** → set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **Long chains OOM** → lower `max_len_per_chain` or crop in the loader.
- **No PDB sequences found** → point `--pdb-root` to real files (symlinks are OK), and use `--verbose` to see per‑file parsing.

---

## Minimal visualization snippets

### Inspect a saved prior (`priors/*.npz`)
```python
import numpy as np, json
from pathlib import Path
npz = np.load(".cache/rescontact/templates/priors/106M_A.npz", allow_pickle=True)
priors = npz["priors"]   # (L, L, B)
bins = npz["bins"]       # (B+1,)
mask = npz["mask"]       # (L, L)
meta = json.loads(str(npz["meta"]))
print(meta, priors.shape, bins.shape, mask.mean())
```

### Quick heatmap of prior mass (sum over bins)
```python
import numpy as np, matplotlib.pyplot as plt
M = priors.sum(-1) * mask
plt.imshow(M, origin="lower"); plt.colorbar(); plt.title("Template prior mass")
plt.show()
```

### Inspect concatenated ESM2+MSA features
```python
import numpy as np
X = np.load("embeddings/esm2_t12_plus_msa/106M_A.concat.npy")
print("Feature shape:", X.shape)  # (L, C + 21 [+1 if depth])
```

---

## License
MIT
