# src/rescontact/api/server.py
from __future__ import annotations

import io, os, math, base64
from pathlib import Path
from typing import List, Optional, Literal, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml

# Biopython (for PDB/mmCIF → sequence)
_HAVE_BIO = True
try:
    from Bio.PDB import PDBParser, MMCIFParser, PPBuilder
except Exception:
    __HAVE_BIO = False

from rescontact.features.esm import ESMEmbedder


# ── helpers ───────────────────────────────────────────────────────────────────

def pick_device(pref: List[str]) -> torch.device:
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
        pad_t = x.new_zeros(*x.shape[:-1], pad)
        return torch.cat([x, pad_t], dim=-1)
    return x[..., :want]

def upper_triangle_indices(L: int, device: torch.device) -> torch.Tensor:
    iu = torch.triu_indices(L, L, offset=1)
    if iu.device != device:
        iu = iu.to(device)
    return iu

def fasta_to_seq_text(fasta_text: str) -> str:
    lines = [ln.strip() for ln in fasta_text.splitlines() if ln.strip()]
    return "".join(ln for ln in lines if not ln.startswith(">")).upper()

def fasta_to_seq_file(fasta_path: str) -> str:
    p = Path(fasta_path)
    if not p.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    seq_lines, seen = [], False
    for ln in p.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(">"):
            if seen:
                break
            seen = True
            continue
        if seen:
            seq_lines.append(s)
    seq = "".join(seq_lines).upper()
    if not seq:
        raise ValueError(f"No sequence read from FASTA: {fasta_path}")
    return seq

def pdb_to_seq(pdb_path: str, chain_id: Optional[str] = "A") -> str:
    if not _HAVEBIO():
        raise RuntimeError("Biopython not installed; run: pip install biopython")
    parser = MMCIFParser(QUIET=True) if pdb_path.lower().endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    struct = parser.get_structure("X", pdb_path)
    model = next(struct.get_models())
    chain = None
    if chain_id:
        for c in model.get_chains():
            if c.id == chain_id:
                chain = c
                break
        if chain is None:
            raise ValueError(f"Chain '{chain_id}' not found in {pdb_path}")
    else:
        chain = next(model.get_chains())
    ppb = PPBuilder()
    peptides = ppb.build_peptides(chain)
    if not peptides:
        raise ValueError("No polypeptide could be built from the selected chain.")
    seq = "".join(str(pp.get_sequence()) for pp in peptides).replace("X", "")
    if not seq:
        raise ValueError("Empty sequence after filtering non-standard residues.")
    return seq

def _HAVEBIO() -> bool:
    return _HAVE_BIO

def to_npz_b64(**arrs) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── tiny head (matches train.py) ──────────────────────────────────────────────

class SimpleContactNet(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dist_bias_max: int = 512, dropout_p: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout_p)
        self.bilin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dist_bias = nn.Embedding(dist_bias_max, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: [B, L, D]
        z = self.drop(self.act(self.proj(emb)))  # [B,L,H]
        zW = self.bilin(z)
        logits = torch.einsum("blh,bmh->blm", zW, z) / math.sqrt(z.shape[-1])  # [B,L,L]
        B, L, _ = logits.shape
        idx = torch.arange(L, device=logits.device)
        dist = (idx[None, :] - idx[:, None]).abs().clamp_max(self.dist_bias.num_embeddings - 1)
        db = self.dist_bias(dist)[:, :, 0]  # [L,L]
        logits = logits + db.unsqueeze(0)
        return logits


# ── config & singletons ──────────────────────────────────────────────────────

CFG_PATH = os.environ.get("RESCONTACT_CONFIG", "configs/rescontact.yaml")
CFG = yaml.safe_load(open(CFG_PATH))

DEVICE = pick_device(CFG["project"].get("device_preference", ["cuda", "mps", "cpu"]))
CACHE_DIR = Path(CFG["paths"]["cache_dir"])
CKPT_DIR  = Path(CFG["paths"]["ckpt_dir"])

ESM_NAME   = CFG["model"]["esm_model"]
EMBED_DIM  = int(CFG["model"]["embed_dim"])
HIDDEN_DIM = int(CFG["model"]["hidden_dim"])
DIST_MAX   = int(CFG["model"]["distance_bias_max"])
DROPOUT_P  = float(CFG["model"].get("dropout_p", 0.1))

_EMBEDDER = ESMEmbedder(ESM_NAME, cache_dir=str(CACHE_DIR), device=DEVICE)

_MODEL = SimpleContactNet(EMBED_DIM, HIDDEN_DIM, DIST_MAX, DROPOUT_P).to(DEVICE).eval()
_ckpt_info: Dict[str, str] = {}
ckpt_path = CKPT_DIR / "model_best.pt"
if ckpt_path.exists():
    try:
        payload = torch.load(str(ckpt_path), map_location=DEVICE)
        state = payload.get("model", payload)
        _MODEL.load_state_dict(state, strict=False)
        _ckpt_info = {"loaded": "true", "path": str(ckpt_path)}
    except Exception as e:
        _ckpt_info = {"loaded": "false", "error": repr(e), "path": str(ckpt_path)}
else:
    _ckpt_info = {"loaded": "false", "reason": "not_found", "path": str(ckpt_path)}


# ── api ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Res-Contact API", version="1.3.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class PredictRequest(BaseModel):
    # One of the following:
    sequence: Optional[str] = Field(None, description="Raw amino-acid sequence (uppercase letters)")
    fasta_text: Optional[str] = Field(None, description="FASTA text (first sequence is used)")
    fasta_path: Optional[str] = Field(None, description="Path to local FASTA file (server-side)")
    pdb_path: Optional[str] = Field(None, description="Path to local PDB/mmCIF file (server-side)")
    chain_id: Optional[str] = Field("A", description="Chain ID when using PDB/mmCIF")

    threshold: float = 0.5
    return_format: Literal["json", "npz"] = "json"
    topk: Optional[int] = Field(None, description="Top-K upper-triangle pairs to return in JSON (default L)")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE.type,
        "esm_model": ESM_NAME,
        "embed_dim": EMBED_DIM,
        "ckpt": _ckpt_info,
    }

def _prepare_sequence(req: PredictRequest) -> str:
    if req.sequence:
        seq = req.sequence.strip().upper()
    elif req.fasta_text:
        seq = fasta_to_seq_text(req.fasta_text)
    elif req.fasta_path:
        seq = fasta_to_seq_file(req.fasta_path)
    elif req.pdb_path:
        try:
            seq = pdb_to_seq(req.pdb_path, req.chain_id or "A")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDB parsing failed: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide one of: sequence | fasta_text | fasta_path | pdb_path.")
    if not seq:
        raise HTTPException(status_code=400, detail="Empty sequence.")
    return seq

@app.post("/predict")
def predict(req: PredictRequest):
    # 1) sequence
    seq = _prepare_sequence(req)
    L = len(seq)

    # 2) embeddings (frozen ESM2)
    try:
        h = _EMBEDDER.embed(seq)  # numpy [L,D0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ESM embedding failed: {e}")

    # 3) forward
    with torch.no_grad():
        x = torch.from_numpy(h).to(DEVICE).unsqueeze(0).float()  # [1,L,D0]
        x = align_embed_dim(x, EMBED_DIM)                         # [1,L,D]
        logits = _MODEL(x)                                        # [1,L,L]
        probs = torch.sigmoid(logits)[0]                          # [L,L]
        probs = 0.5 * (probs + probs.T)                           # symmetrize
        probs.fill_diagonal_(0.0)
        thr = float(req.threshold)
        binary = (probs >= thr).to(torch.uint8)

    if req.return_format == "npz":
        return {
            "L": L,
            "threshold": thr,
            "npz_b64": to_npz_b64(
                probs=probs.detach().cpu().numpy(),
                binary=binary.detach().cpu().numpy()
            ),
        }

    # JSON: top-K upper-triangle pairs
    iu = upper_triangle_indices(L, probs.device)
    scores = probs[iu[0], iu[1]].detach().cpu().numpy()
    preds  = binary[iu[0], iu[1]].detach().cpu().numpy()

    K = int(req.topk) if req.topk is not None else L
    K = max(1, min(K, scores.shape[0]))

    if K < scores.shape[0]:
        idx_k = np.argpartition(-scores, K-1)[:K]
        idx_sorted = idx_k[np.argsort(-scores[idx_k])]
    else:
        idx_sorted = np.argsort(-scores)

    pairs = [
        {
            "i": int(iu[0][idx].item()),
            "j": int(iu[1][idx].item()),
            "score": float(scores[idx]),
            "pred": bool(int(preds[idx])),
        }
        for idx in idx_sorted[:K]
    ]

    return {
        "L": L,
        "threshold": thr,
        "topk": K,
        "pairs": pairs,
        "meta": {
            "device": DEVICE.type,
            "esm_model": ESM_NAME,
            "embed_dim": EMBED_DIM,
            "ckpt_loaded": _ckpt_info.get("loaded", "false"),
        },
    }

@app.get("/")
def root():
    return {"message": "Res-Contact API is up. POST /predict with sequence|fasta_text|fasta_path|pdb_path."}
