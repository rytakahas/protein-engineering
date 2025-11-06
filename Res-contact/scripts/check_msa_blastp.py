#!/usr/bin/env python3
import time
from time import perf_counter
import yaml
import torch
from pathlib import Path
import sys

# --- add ./src to sys.path so "rescontact" imports work ---
ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ----------------------------------------------------------

from rescontact.data.dataset import PDBContactDataset

# Load config
cfg = yaml.safe_load(open("configs/rescontact.yaml"))

# Build dataset (uses your YAML settings)
ds = PDBContactDataset(
    root_dir=cfg["paths"]["train_dir"],
    cache_dir=cfg["paths"]["cache_dir"],
    contact_threshold=cfg["labels"]["contact_threshold_angstrom"],
    include_inter_chain=cfg["labels"]["include_inter_chain"],
    esm_model_name=cfg["model"]["esm_model"],
    use_msa=cfg["features"]["use_msa"],
    msa_cfg=cfg["features"]["msa"],
)

print(f"dataset size: {len(ds)}")

if len(ds) == 0:
    raise SystemExit("No examples found. Check paths.train_dir in configs/rescontact.yaml.")

# Load one example first (kept from original)
t0 = time.time()
it = ds[0]  # first example
dt = time.time() - t0
print(f"loaded first example in {dt:.2f}s; id={it.get('id','?')}")

emb_t = it["emb"]
# Ensure CPU numpy
if isinstance(emb_t, torch.Tensor):
    emb = emb_t.detach().cpu().numpy()
else:
    emb = emb_t

L, D = emb.shape
print("emb shape:", (L, D))
print("msa_path:", it.get("msa_path"))

# Only interpret "last 21 dims" if we know D>=341 (ESM+MSA layout)
if D >= 341:
    last21 = emb[:, -21:]
    nonzero = int((last21 != 0).sum())
    print("nonzero in last-21 dims:", nonzero,
          "(>0 means BLAST/Jackhmmer/local MSA present; 0 means zero-padded)")
else:
    print("No MSA concatenated (D < 341). If you expect MSA, set features.use_msa: true "
          "and ensure your dataset.py pads zeros when providers are unavailable.")

# --- Extra: probe first up-to-3 items with precise timings ---
N = min(10, len(ds))
for i in range(N):
    t0 = perf_counter()
    itm = ds[i]
    dt = perf_counter() - t0
    emb_i = itm["emb"]
    if isinstance(emb_i, torch.Tensor):
        emb_np = emb_i.detach().cpu().numpy()
    else:
        emb_np = emb_i
    Di = emb_np.shape[1]
    nz21 = int((emb_np[:, -21:] != 0).sum()) if Di >= 341 else -1
    print(f"[{i}] id={itm.get('id','?')}  load={dt:.2f}s  D={Di}  "
          f"msa_nonzero={nz21}  msa_path={itm.get('msa_path')}")

