# ResIntNet (lightweight package)

A compact, Mac‑friendly implementation of **Residual Interaction Network with Adaptive Memory** (g_mem).
Runs on an 8GB MacBook Air (M3) with CPU/MPS. Downloads PDB/AFDB structures on demand.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```
(For pretty progress bars) `pip install ipywidgets`

## Quickstart
```bash
python scripts/train_memory.py --terms TP53 LYSOZYME --n_structures 20 --train_steps 200 --out ./outputs
python scripts/rank_mutations.py --out ./outputs --topk 10
```
Per‑variant DMS CSV (pos, wt, mut, score): `scripts/ingest_mutations.py`

Edit `resintnet/ingest/__init__.py` to wire your real DMS/FireProt loaders.
