Res-Contact

Lightweight, end-to-end pipeline for protein contact map prediction on an 8 GB MacBook Air M3.
It builds 8 Å ground-truth (Cβ; Gly→Cα) from PDB/mmCIF, embeds sequences with tiny ESM2, trains a bilinear distance-biased model, and serves predictions via FastAPI.
MSA is optional with graceful fallbacks: local → jackhmmer → blastp → skip.

1) Repo layout
.
├─ configs/
│  └─ rescontact.yaml          # all config (paths, model, training, api, msa)
├─ scripts/
│  ├─ train.py                 # train on data/pdb/train (splits 0.8/0.2)
│  └─ eval.py                  # evaluate on data/pdb/test (or train)
├─ src/rescontact/
│  ├─ __init__.py
│  ├─ api/
│  │  ├─ __init__.py
│  │  └─ server.py             # FastAPI server (/predict)
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ dataset.py            # PDB→labels, ESM cache, MSA path detection
│  │  └─ pdb_utils.py          # PDB/mmCIF parsing, 8Å contact labels
│  ├─ features/
│  │  ├─ __init__.py
│  │  ├─ embedding.py          # tiny ESM2 embeddings (+disk cache)
│  │  ├─ msa.py                # MSA fallbacks (local/jackhmmer/blastp/skip)
│  │  └─ pair_features.py      # (placeholder) distance buckets, etc.
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ contact_net.py        # bilinear scoring + distance bias
│  └─ utils/
│     ├─ __init__.py
│     ├─ metrics.py            # P@L, ROC-AUC, F1
│     └─ train.py              # device, seeding, early stop, ckpt I/O
├─ tests/                      # minimal smoke tests
│  ├─ test_msa_providers_mock.py
│  ├─ test_pair_features.py
│  ├─ test_pdb_utils.py
│  └─ test_train_smoke.py
├─ requirements.txt
└─ roadmap.txt

Data folders (you create them)
data/
├─ pdb/
│  ├─ train/   # *.pdb or *.cif or *.mmCIF
│  └─ test/    # *.pdb or *.cif or *.mmCIF
└─ msa/
   └─ ...      # optional local *.a3m files (matched by glob in config)

Cache & outputs (auto-created)
.cache/rescontact/           # ESM NPZ cache (per sequence)
.cache/rescontact/msatmp/    # temp FASTA when querying jackhmmer/blastp
checkpoints/                 # saved model weights
logs/                        # (placeholder)

2) Installation (Mac, Python 3.10–3.12)
# from repo root
python -m venv .venv
source .venv/bin/activate

# PyTorch first (CPU/MPS build; MPS is included in macOS wheels)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Rest of deps (ESM, FastAPI, etc.)
pip install -r requirements.txt


Note (ESM): fair-esm pulls model weights at first use and we cache embeddings to disk to keep RAM low.

3) Configure

Edit configs/rescontact.yaml if needed:

paths.train_dir / test_dir: where your PDB/mmCIF live

labels.contact_threshold_angstrom: 8.0 by default

features.use_msa: true tries MSA; missing tools/DBs are silently skipped

features.msa.local_glob: pattern for local *.a3m

features.msa.jackhmmer / blastp: set binary and db if available

model.esm_model: esm2_t6_8M_UR50D (tiny)

api: FastAPI host/port

4) Prepare data

Place training structures under data/pdb/train/ and test structures under data/pdb/test/:

data/pdb/train/1abc_A.pdb
data/pdb/train/2xyz.cif
data/pdb/test/ 3def_A+B.cif


Ground truth: computed on the fly as 8 Å Cβ–Cβ (Gly→Cα) distance map (symmetric, diagonal zeros).
Multimers: we yield both intra-chain and, if enabled, inter-chain examples (monomer/dimer/multimer handled automatically).

Optional local MSAs:

data/msa/myprotein_XXXX.a3m


(We just detect presence; v0.1 does not yet feed MSA into the model.)

5) Train (0.8/0.2 split on train set)
python scripts/train.py --config configs/rescontact.yaml


Uses tiny ESM2; embeddings cached under .cache/rescontact.

Batch size = 1 to avoid L×L padding blow-ups (M3-friendly).

Mixed precision: disabled on CPU and MPS autocast; handled automatically.

Early stopping on validation loss.

Sample log per epoch:

[epoch 3] train=0.4912  val=0.5053  P@L=0.417  ROC=0.731  F1=0.544

6) Evaluate (on test set)
python scripts/eval.py \
  --config configs/rescontact.yaml \
  --ckpt checkpoints/model_best.pt \
  --split test


Outputs JSON with P@L, ROC-AUC, F1@threshold.

7) Inference
A) FastAPI server (recommended)

Start server:

uvicorn rescontact.api.server:app --host 0.0.0.0 --port 8000


Predict from a raw sequence (FASTA-like string):

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence":"ACDEFGHIKLMNPQRSTVWY"}' \
  | jq .


Predict from a PDB/mmCIF path (server reads sequence from structure):

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pdb_path":"data/pdb/test/3def_A+B.cif","threshold":0.5}' \
  | jq .


The response contains a base64-encoded NPZ with:

probs : L×L probabilities

binary: L×L uint8 (thresholded)

Decode the NPZ (Python client):

import base64, io, numpy as np, requests
r = requests.post("http://localhost:8000/predict", json={"sequence":"ACDEFGHIKLMNPQRSTVWY"})
b = base64.b64decode(r.json()["npz_b64"])
npz = np.load(io.BytesIO(b))
probs, binary = npz["probs"], npz["binary"]
print(probs.shape, binary.sum())

B) Direct Python (no server)
import torch, numpy as np
from rescontact.features.embedding import ESMEmbedder
from rescontact.models.contact_net import BilinearContactNet

seq = "ACDEFGHIKLMNPQRSTVWY"
embedder = ESMEmbedder("esm2_t6_8M_UR50D", "./.cache/rescontact", "mps" if torch.backends.mps.is_available() else "cpu")
emb = torch.from_numpy(embedder.embed(seq))
model = BilinearContactNet(embed_dim=320, hidden_dim=256, distance_bias_max=512)
ckpt = torch.load("checkpoints/model_best.pt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"]); model.eval()
with torch.no_grad():
    logits = model(emb)
    probs = torch.sigmoid(logits).numpy()
binary = (probs >= 0.5).astype(np.uint8)
print(probs.shape, binary.sum())

8) MSA (optional, safe fallbacks)

Config: features.use_msa: true
Order: local .a3m → jackhmmer → blastp → skip

Local: put files under data/msa/ matching local_glob (e.g., **/*.a3m).

jackhmmer: set binary and db in YAML; if missing, skipped.

blastp: same; if unavailable, skipped.

v0.1 only detects availability and stores the path. Integration into model features is planned for v0.2.

9) Mac M3 tips (8 GB)

Keep batch_size=1 (already default).

Use tiny ESM2 (esm2_t6_8M_UR50D).

Inter-chain contacts increase L; if memory is tight, set
labels.include_inter_chain: false.

Embeddings are cached; subsequent runs are much lighter.

If you see MPS oddities, set training.mixed_precision: false in YAML.

10) Troubleshooting

Vim E212: Can't open file for writing: parent folder missing or not writable.
mkdir -p src/rescontact/api && chmod u+w src/rescontact/api

fair-esm import error: ensure pip install fair-esm (it’s in requirements.txt).

Slow first epoch: ESM embeddings are computed and cached; later epochs reuse cache.

Model length mismatch: some PDBs have missing CB/CA → NaN rows are masked.
We assert ESM length equals concatenated sequence length; if you edit PDBs, keep residues aligned.

11) Config cheat-sheet
labels:
  contact_threshold_angstrom: 8.0    # 8 Å contacts
  include_inter_chain: true          # inter-chain labels on/off
training:
  epochs: 20
  batch_size: 1
  lr: 1.5e-3
  patience: 5
model:
  esm_model: esm2_t6_8M_UR50D
  embed_dim: 320
  hidden_dim: 256
features:
  use_msa: true
  msa:
    local_glob: data/msa/**/*.a3m
    jackhmmer: {enabled: true, binary: jackhmmer, db: null}
    blastp:    {enabled: true, binary: blastp,    db: null}
api:
  framework: fastapi
  host: 0.0.0.0
  port: 8000

12) Run end-to-end quickstart
# 0) setup
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 1) put PDBs
mkdir -p data/pdb/train data/pdb/test
# copy your *.pdb / *.cif files accordingly

# 2) train (splits 0.8/0.2 of train internally)
python scripts/train.py --config configs/rescontact.yaml

# 3) evaluate on test
python scripts/eval.py --config configs/rescontact.yaml --ckpt checkpoints/model_best.pt --split test

# 4A) serve API
uvicorn rescontact.api.server:app --host 0.0.0.0 --port 8000
# curl or Python client as shown above

# 4B) OR: run direct Python for a sequence (no server)
# (see snippet in §7B)

