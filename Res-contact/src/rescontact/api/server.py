# src/rescontact/api/server.py
import os, io, base64, math
from pathlib import Path
from typing import Optional, Tuple, List

# Use a headless backend for server-side plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------
# Tiny utils
# -----------------------
def pick_device(pref: list[str]) -> torch.device:
    for p in pref:
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if p == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")

def align_embed_dim(x: torch.Tensor, want: int) -> torch.Tensor:
    got = x.shape[-1]
    if got == want:
        return x
    if got < want:
        pad = want - got
        return torch.cat([x, x.new_zeros(*x.shape[:-1], pad)], dim=-1)
    return x[..., :want]

def load_fasta_first(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")
    seq_lines: List[str] = []
    reading = False
    with p.open() as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if reading:
                    break
                reading = True
                continue
            if reading:
                seq_lines.append(s)
    seq = "".join(seq_lines).upper()
    if not seq:
        raise ValueError(f"No sequence read from FASTA: {path}")
    return seq

def sequence_from_pdb_quick(path: str) -> str:
    """
    VERY lightweight sequence extraction:
    - Uses biopython if available.
    - Concatenates first model's first chain residues; non-std aa skipped.
    Good enough for quick inference; training should keep the dataset pipeline.
    """
    try:
        from Bio.PDB import MMCIFParser, PDBParser, PPBuilder
        ppb = PPBuilder()
        parser = MMCIFParser(QUIET=True) if path.lower().endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
        struct = parser.get_structure("X", path)
        # Take first model, first chain
        model = next(struct.get_models())
        chain = next(model.get_chains())
        # Build polypeptides then join
        pps = ppb.build_peptides(chain)
        if not pps:
            raise ValueError("No polypeptide built from chain.")
        seq = "".join([pp.get_sequence().__str__() for pp in pps])
        seq = seq.replace("X", "")  # drop unknowns
        if not seq:
            raise ValueError("Empty sequence after filtering.")
        return seq
    except Exception as e:
        raise RuntimeError(f"PDB parse failed: {e}")

# -----------------------
# Model / Embedder
# -----------------------
# Prefer your project ESM embedder + model
from rescontact.features.embedding import ESMEmbedder

try:
    from rescontact.models.contact_net import BilinearContactNet as ContactNet
except Exception:
    # Fallback: simple contact net (matches training script shape)
    class ContactNet(nn.Module):
        def __init__(self, embed_dim: int, hidden_dim: int, distance_bias_max: int = 512):
            super().__init__()
            self.proj = nn.Linear(embed_dim, hidden_dim)
            self.act = nn.ReLU()
            self.bilin = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.dist_bias = nn.Embedding(distance_bias_max, 1)

        def forward(self, emb: torch.Tensor) -> torch.Tensor:
            z = self.act(self.proj(emb))
            zW = self.bilin(z)
            logits = torch.einsum("blh,bmh->blm", zW, z) / math.sqrt(z.shape[-1])
            B, L, _ = logits.shape
            idx = torch.arange(L, device=logits.device)
            dist = (idx[None, :] - idx[:, None]).abs().clamp_max(self.dist_bias.num_embeddings - 1)
            db = self.dist_bias(dist)[:, :, 0]
            return logits + db.unsqueeze(0)

# Load YAML config
CFG_PATH = os.environ.get("RESCONTACT_CONFIG", "configs/rescontact.yaml")
CFG = yaml.safe_load(open(CFG_PATH))

DEVICE = pick_device(CFG["project"].get("device_preference", ["cuda", "mps", "cpu"]))
CACHE = Path(CFG["paths"]["cache_dir"])
CACHE.mkdir(parents=True, exist_ok=True)
CKPT = Path(CFG["paths"]["ckpt_dir"]) / "model_best.pt"

ESM_NAME = CFG["model"]["esm_model"]
WANT_DIM = int(CFG["model"]["embed_dim"])
HID_DIM = int(CFG["model"]["hidden_dim"])
DIST_MAX = int(CFG["model"]["distance_bias_max"])

# Initialize embedder + model once
_EMBEDDER = ESMEmbedder(ESM_NAME, cache_dir=str(CACHE), device=DEVICE)
_MODEL = ContactNet(embed_dim=WANT_DIM, hidden_dim=HID_DIM, distance_bias_max=DIST_MAX).to(DEVICE).eval()

if CKPT.exists():
    state = torch.load(CKPT, map_location=DEVICE)
    # Accept either {"model":...} or {"state_dict":...}
    weights = state.get("model", state.get("state_dict"))
    if weights is not None:
        _MODEL.load_state_dict(weights, strict=False)
else:
    print(f"[server] WARNING: checkpoint not found at {CKPT}; using randomly initialized weights")

# -----------------------
# Inference helpers
# -----------------------
@torch.no_grad()
def predict_probs_from_sequence(seq: str) -> np.ndarray:
    """Returns LxL probabilities (numpy)."""
    seq = seq.strip().upper()
    if not seq:
        raise ValueError("Empty sequence.")
    # ESM embed
    if hasattr(_EMBEDDER, "embed"):
        emb_np = _EMBEDDER.embed(seq)  # shape [L, D]
    else:
        # Back-compat if your class exposes a different name
        emb_np = _EMBEDDER.embed_sequence(seq)
    emb = torch.from_numpy(emb_np).unsqueeze(0).float().to(DEVICE)  # [1,L,D]
    emb = align_embed_dim(emb, WANT_DIM)  # pad/truncate for (with/without MSA)
    logits = _MODEL(emb).squeeze(0)  # [L,L]
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs

def _npz_b64_from_probs(probs: np.ndarray, threshold: float = 0.5) -> str:
    binary = (probs >= threshold).astype(np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, probs=probs, binary=binary)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -----------------------
# FastAPI
# -----------------------
app = FastAPI(title="ResContact API", version="0.1.0")

class PredictReq(BaseModel):
    sequence: Optional[str] = None
    fasta_path: Optional[str] = None
    pdb_path: Optional[str] = None
    threshold: float = 0.5

class PredictResp(BaseModel):
    L: int
    threshold: float
    npz_b64: str

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE.type, "embed_dim": WANT_DIM}

@app.post("/predict", response_model=PredictResp)
def predict(req: PredictReq):
    try:
        if req.sequence:
            seq = req.sequence
        elif req.fasta_path:
            seq = load_fasta_first(req.fasta_path)
        elif req.pdb_path:
            seq = sequence_from_pdb_quick(req.pdb_path)
        else:
            raise HTTPException(status_code=400, detail="Provide one of: sequence | fasta_path | pdb_path")

        probs = predict_probs_from_sequence(seq)
        b64 = _npz_b64_from_probs(probs, threshold=float(req.threshold))
        return {"L": int(probs.shape[0]), "threshold": float(req.threshold), "npz_b64": b64}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Visualization endpoint
# -----------------------
class VizReq(BaseModel):
    sequence: Optional[str] = None
    fasta_path: Optional[str] = None
    pdb_path: Optional[str] = None
    threshold: float = 0.5

class VizResp(BaseModel):
    L: int
    png_b64: str

@app.post("/visualize", response_model=VizResp)
def visualize(req: VizReq):
    try:
        if req.sequence:
            seq = req.sequence
        elif req.fasta_path:
            seq = load_fasta_first(req.fasta_path)
        elif req.pdb_path:
            seq = sequence_from_pdb_quick(req.pdb_path)
        else:
            raise HTTPException(status_code=400, detail="Provide one of: sequence | fasta_path | pdb_path")

        probs = predict_probs_from_sequence(seq)
        L = probs.shape[0]

        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(probs, vmin=0, vmax=1)
        ax.set_title(f"Contact prob (L={L})")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"L": int(L), "png_b64": b64}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

