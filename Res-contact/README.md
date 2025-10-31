### Res-Contact
===========

End-to-end pipeline for protein contact map prediction.

It builds 8 Å ground-truth (Cβ; Gly→Cα) from PDB/mmCIF, embeds sequences with tiny ESM2, trains a bilinear distance-biased model, and serves predictions via FastAPI.

MSA is optional with graceful fallbacks: local → jackhmmer → blastp → skip.

---

#### 1) Repo layout

```
├─ configs/
│ └─ rescontact.yaml # all config (paths, model, training, api, msa, monitoring)
├─ monitor/
│ └─ baseline.json # (generated) PSI baseline (bin edges + base proportions)
├─ scripts/
│ ├─ train.py # train on data/pdb/train (splits 0.8/0.2)
│ ├─ eval.py # evaluate on data/pdb/test (or train)
│ └─ build_baseline.py # build PSI baseline (seq_len, prob_scores, etc.)
├─ src/rescontact/
│ ├─ init.py
│ ├─ api/
│ │ ├─ init.py
│ │ └─ server.py # FastAPI server (/predict, /visualize, /psi, /metrics, /admin/reset_psis)
│ ├─ data/
│ │ ├─ init.py
│ │ ├─ dataset.py # PDB→labels, ESM cache, MSA path detection
│ │ └─ pdb_utils.py # PDB/mmCIF parsing, 8Å contact labels
│ ├─ features/
│ │ ├─ init.py
│ │ ├─ embedding.py # tiny ESM2 embeddings (+disk cache)
│ │ ├─ msa.py # MSA fallbacks (local/jackhmmer/blastp/skip)
│ │ └─ pair_features.py # (placeholder) distance buckets, etc.
│ ├─ models/
│ │ ├─ init.py
│ │ └─ contact_net.py # bilinear scoring + distance bias
│ └─ utils/
│ ├─ init.py
│ ├─ metrics.py # P@L, ROC-AUC, F1
│ └─ train.py # device, seeding, early stop, ckpt I/O
├─ tests/ # minimal smoke tests
│ ├─ test_msa_providers_mock.py
│ ├─ test_pair_features.py
│ ├─ test_pdb_utils.py
│ └─ test_train_smoke.py
├─ requirements.txt
└─ roadmap.txt
```

#### Data folders (you create them)

```
data/
├─ pdb/
│ ├─ train/ # *.pdb or *.cif or *.mmCIF
│ └─ test/ # *.pdb or *.cif or *.mmCIF
└─ msa/
└─ ... # optional local *.a3m files (matched by glob in config)
```
#### Cache & outputs (auto-created)

```
.cache/rescontact/ # ESM NPZ cache (per sequence)
.cache/rescontact/msatmp/ # temp FASTA when querying jackhmmer/blastp
checkpoints/ # saved model weights
logs/ # (placeholder)
```

---

#### 2) Installation (Mac, Python 3.10–3.12)

> From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
Install PyTorch (CPU/MPS build; MPS ships with macOS wheels):
```

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
Install remaining deps:
```

```bash
pip install -r requirements.txt
```
Note (ESM): fair-esm pulls model weights at first use. Embeddings are cached on disk.

3) Configure
Edit configs/rescontact.yaml if needed:

paths.train_dir / paths.test_dir: where PDB(/mmCIF) live

labels.contact_threshold_angstrom: 8.0 by default

features.use_msa: true tries MSA; missing tools/DBs are skipped gracefully

features.msa.local_glob: pattern for local *.a3m

features.msa.jackhmmer / blastp: set provider params if available

model.esm_model: facebook/esm2_t6_8M_UR50D (tiny)

api: FastAPI host/port

Monitoring (PSI):

```yaml
Copiar código
monitoring:
  enabled: true
  drift_method: "psi"
  psi_bins: 10
  psi_warn: 0.10
  psi_alert: 0.20
  baseline_path: "monitor/baseline.json"
  features:
    - name: "seq_len"      # numeric
      kind: "numeric"
    - name: "prob_scores"  # L×L upper triangle contact probabilities
      kind: "numeric"
    - name: "pos_distance" # |i-j| among predicted positives
      kind: "numeric"
    - name: "emb_norms"    # per-residue embedding norms
      kind: "numeric"
    - name: "msa_coverage" # fraction of non-zero in last 21 dims (MSA)
      kind: "numeric"
```

4) Prepare data
Place training structures under data/pdb/train/ and test structures under data/pdb/test/:

```bash
data/pdb/train/1abc_A.pdb
data/pdb/train/2xyz.cif
data/pdb/test/3def_A+B.cif
```
Ground truth is computed on the fly as 8 Å Cβ–Cβ (Gly→Cα) distance map (symmetric, diagonal zeros).
Multimers: both intra-chain and (if enabled) inter-chain examples are handled (monomer/dimer/multimer).

Optional local MSAs:

```bash
data/msa/myprotein_XXXX.a3m
```
v0.1 detects MSA presence; MSA feature integration is optional and budgeted.

5) Train (0.8/0.2 split on train set)
```bash
python scripts/train.py --config configs/rescontact.yaml
```
Uses tiny ESM2; embeddings cached under .cache/rescontact

Batch size = 1 to avoid L×L padding blow-ups (M3-friendly)

Mixed precision: disabled automatically on CPU/MPS

Early stopping on validation loss

Sample log:

```ini
[epoch 3] train=0.4912  val=0.5053  P@L=0.417  ROC=0.731  F1=0.544
```
6) Evaluate (on test set)
```bash
python scripts/eval.py \
  --config configs/rescontact.yaml \
  --ckpt checkpoints/model_best.pt \
  --split test
 ```
Outputs JSON with P@L, ROC-AUC, F1@threshold.

7) Inference
A) FastAPI server (recommended)
Start server:

```bash
uvicorn rescontact.api.server:app --host 0.0.0.0 --port 8000
```
Predict from a raw sequence:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence":"ACDEFGHIKLMNPQRSTVWY"}' \
  | jq .
```
Predict from a PDB(/mmCIF) path (server derives sequence):

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pdb_path":"data/pdb/test/3def_A+B.cif","threshold":0.5}' \
  | jq .
  ```
The response contains a base64-encoded NPZ with:

probs : L×L probabilities

binary : L×L uint8 (thresholded)

Decode the NPZ (Python client):

```python
import base64, io, numpy as np, requests
r = requests.post("http://localhost:8000/predict", json={"sequence":"ACDEFGHIKLMNPQRSTVWY"})
b = base64.b64decode(r.json()["npz_b64"])
npz = np.load(io.BytesIO(b))
probs, binary = npz["probs"], npz["binary"]
print(probs.shape, binary.sum())
```
Optional helper to save shapes quickly:

```bash
cat > check_npz.py <<'PY'
import base64, sys, numpy as np, json
b64 = json.loads(sys.stdin.read())["npz_b64"]
with open("preds.npz","wb") as f: f.write(base64.b64decode(b64))
d = np.load("preds.npz")
print("probs", d["probs"].shape, "binary", d["binary"].shape)
PY

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence":"ACDEFGHIKLMNPQRSTVWY"}' | python check_npz.py
```

B) Direct Python (no server)
```python
import torch, numpy as np
from rescontact.features.embedding import ESMEmbedder
from rescontact.models.contact_net import BilinearContactNet

seq = "ACDEFGHIKLMNPQRSTVWY"
device = "mps" if torch.backends.mps.is_available() else "cpu"
embedder = ESMEmbedder("esm2_t6_8M_UR50D", "./.cache/rescontact", device)
emb = torch.from_numpy(embedder.embed(seq))
model = BilinearContactNet(embed_dim=320, hidden_dim=256, distance_bias_max=512)
ckpt = torch.load("checkpoints/model_best.pt", map_location="cpu")
model.load_state_dict(ckpt.get("state_dict") or ckpt.get("model") or ckpt)
model.eval()
with torch.no_grad():
    logits = model(emb)          # supports [L,D] or [1,L,D]
    if logits.ndim == 3: logits = logits.squeeze(0)
    probs = torch.sigmoid(logits).numpy()
binary = (probs >= 0.5).astype(np.uint8)
print(probs.shape, binary.sum())
```

8) Monitoring (PSI drift)
This repo includes a lightweight PSI monitor built into server.py. It tracks distributional drift between a fixed baseline and a live aggregate window.

What’s monitored?
seq_len: distribution of sequence lengths L

prob_scores: distribution of predicted probabilities on the upper triangle of the L×L map

pos_distance: distribution of |i − j| among predicted positives (≥ threshold)

emb_norms: per-residue ESM embedding norms

msa_coverage: fraction of non-zero entries in the last 21 dims (0 for ESM-only inputs)

PSI is computed against your saved baseline; the live window aggregates proportions over requests and reports a value plus a coarse category.

Build a baseline (once)
```bash
PYTHONPATH=src python scripts/build_baseline.py \
  --config configs/rescontact.yaml \
  --out monitor/baseline.json \
  --max_examples 200
```
This creates monitor/baseline.json capturing bin edges (quantile bins) and base proportions for each metric.

Start API
```bash
uvicorn rescontact.api.server:app --host 0.0.0.0 --port 8000
```
Generate traffic
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence":"ACDEFGHIKLMNPQRSTVWY"}' | jq .
```
Inspect PSI / metrics
```bash
curl -s http://localhost:8000/psi | jq .
curl -s http://localhost:8000/metrics
```
/psi returns JSON snapshot:

values and categories for each metric

request/error counters

latency p50/p95 (internal monitor compute)

/metrics returns Prometheus-formatted text, including:

rescontact_requests_total

rescontact_errors_total

rescontact_latency_ms{quantile="0.50"/"0.95"}

rescontact_psi{metric="<name>", category="<stable|slight_shift|...>"}

Reset the live window
```bash
curl -s -X POST http://localhost:8000/admin/reset_psis
```
Baseline stays; only the live aggregate is cleared.

PSI categories (defaults)
Configured in YAML:

psi_warn = 0.10 → stable if PSI < 0.10

psi_alert = 0.20:

[0.10, 0.20) → slight_shift

[0.20, max(2×alert, 0.5)) → moderate_shift

≥ max(2×alert, 0.5) → major_shift

You can tune these thresholds in configs/rescontact.yaml.

Prometheus scrape example
Add to your Prometheus scrape_configs:

```yaml
scrape_configs:
  - job_name: rescontact
    metrics_path: /metrics
    static_configs:
  - targets: ["localhost:8000"]   # or your service DNS
```
Use Grafana to visualize PSI over time per metric and alert on category changes.

Notes & tips
Quantile bins (from baseline) stabilize PSI and avoid brittleness across ranges.

Zeros & logs: proportions are clipped with a small ε to avoid log(0).

Windowing: the default window aggregates over the server lifetime (or since last reset). For sliding windows, adapt the monitor to use a ring buffer or deque of recent vectors.

Per-mode splits: if you serve both ESM and MSA models, either maintain two monitors or tag observations and split them downstream (e.g., in Prometheus labels).

9) MSA (optional, safe fallbacks)
Config: features.use_msa: true
Order: local .a3m → jackhmmer → blastp → skip

Local: put files under data/msa/ matching local_glob (e.g., **/*.a3m)

jackhmmer: set provider params in YAML; if missing/unavailable, skipped

blastp: same; if unavailable, skipped

For real-time serving, keep ESM-only by default; use precomputed MSA features for batch/scoring pipelines.

10) Mac M3 tips (8 GB)
Keep training.batch_size = 1 (default)

Use tiny ESM2 (esm2_t6_8M_UR50D)

Inter-chain contacts increase L; if memory is tight, set labels.include_inter_chain: false

Embeddings are cached; subsequent runs are much lighter

If you see MPS oddities, set training.mixed_precision: false in YAML

11) Troubleshooting
“Vim E212: Can't open file for writing”
Create folders and ensure write perms:

```bash
mkdir -p src/rescontact/api && chmod u+w src/rescontact/api
```
fair-esm import error: ensure pip install fair-esm (included in requirements.txt)

Slow first epoch: ESM embeddings are computed and cached; later epochs reuse cache

Model tensor shape issues:
Serving path accepts both [L,D] and [1,L,D]. The API’s model wrapper avoids .t() on 3-D tensors.

PSI shows “major_shift” for pos_distance on toy runs:
That’s expected on very small or synthetic sequences; build the baseline from representative data.

12) Config cheat-sheet
```yaml
labels:
  contact_threshold_angstrom: 8.0
  include_inter_chain: true

training:
  epochs: 20
  batch_size: 1
  lr: 1.5e-3
  patience: 5

model:
  esm_model: facebook/esm2_t6_8M_UR50D
  embed_dim: 320
  hidden_dim: 256
  distance_bias_max: 512

features:
  use_msa: false
  msa:
    local_glob: data/msa/**/*.a3m
    jackhmmer_remote:
      enabled: true
      db: uniprotrefprot
      email: you@example.com
      timeout_s: 600
      poll_every_s: 5
      max_seqs: 64
    blastp:
      enabled: true
      db: swissprot
      hitlist_size: 64
      expect: "1e-2"
      timeout_s: 600
      max_seqs: 64
    max_seqs: 64

api:
  framework: fastapi
  host: 0.0.0.0
  port: 8000

monitoring:
  enabled: true
  drift_method: psi
  psi_bins: 10
  psi_warn: 0.10
  psi_alert: 0.20
  baseline_path: monitor/baseline.json
```
13) Run end-to-end quickstart
0) setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
1) put PDBs
```bash
mkdir -p data/pdb/train data/pdb/test
# copy your *.pdb / *.cif files accordingly
```
2) train (splits 0.8/0.2 of train internally)
```bash
python scripts/train.py --config configs/rescontact.yaml
```
3) evaluate on test
```bash
python scripts/eval.py --config configs/rescontact.yaml --ckpt checkpoints/model_best.pt --split test
```
4A) serve API
```bash
uvicorn rescontact.api.server:app --host 0.0.0.0 --port 8000
```
4B) monitor (PSI)
```bash
# build baseline once
PYTHONPATH=src python scripts/build_baseline.py \
  --config configs/rescontact.yaml \
  --out monitor/baseline.json \
  --max_examples 200

# generate some requests
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence":"ACDEFGHIKLMNPQRSTVWY"}' | jq .

# inspect PSI and metrics
curl -s http://localhost:8000/psi | jq .
curl -s http://localhost:8000/metrics

# reset live window
curl -s -X POST http://localhost:8000/admin/reset_psis
```
14) Production notes
Online inference: serve ESM-only; use MSA only if precomputed features exist

Batch pipelines: run Jackhmmer/BLAST locally and persist A3M/feature NPZs

Model zoo: keep both checkpoints/model_esm.pt and checkpoints/model_msa.pt

Cache hygiene: maintain separate caches .cache/rescontact_esm/ vs .cache/rescontact_msa/; add LRU/TTL cleanup

Observability: scrape /metrics with Prometheus; alert on PSI category or value
