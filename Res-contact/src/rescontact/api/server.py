# src/rescontact/api/server.py
import os, io, base64, math, time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Headless plotting for server-side figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import yaml
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# -----------------------
# Utilities
# -----------------------
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
    """Pad/slice last-dim to 'want'. Accept [L,D] or [B,L,D]."""
    if x.dim() == 2:
        L, D = x.shape
        if D == want: return x
        if D < want:  return torch.cat([x, x.new_zeros(L, want - D)], dim=-1)
        return x[:, :want]
    if x.dim() == 3:
        B, L, D = x.shape
        if D == want: return x
        if D < want:  return torch.cat([x, x.new_zeros(B, L, want - D)], dim=-1)
        return x[:, :, :want]
    raise ValueError(f"Expected [L,D] or [B,L,D], got {tuple(x.shape)}")

def load_fasta_first(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")
    seq_lines: List[str] = []
    reading = False
    with p.open() as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if s.startswith(">"):
                if reading: break
                reading = True
                continue
            if reading: seq_lines.append(s)
    seq = "".join(seq_lines).upper()
    if not seq:
        raise ValueError(f"No sequence read from FASTA: {path}")
    return seq

def sequence_from_pdb_quick(path: str) -> str:
    """Quick sequence extraction from first model/chain using Bio.PDB."""
    try:
        from Bio.PDB import MMCIFParser, PDBParser, PPBuilder
        parser = MMCIFParser(QUIET=True) if path.lower().endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
        struct = parser.get_structure("X", path)
        model = next(struct.get_models())
        chain = next(model.get_chains())
        pps = PPBuilder().build_peptides(chain)
        if not pps: raise ValueError("No polypeptide built from chain.")
        seq = "".join([pp.get_sequence().__str__() for pp in pps]).replace("X", "")
        if not seq: raise ValueError("Empty sequence after filtering.")
        return seq
    except Exception as e:
        raise RuntimeError(f"PDB parse failed: {e}")

# -----------------------
# Model / Embedder
# -----------------------
from rescontact.features.embedding import ESMEmbedder

class SimpleContactNet(nn.Module):
    """Safe fallback contact head—no .t() on 3-D anywhere."""
    def __init__(self, embed_dim: int, hidden_dim: int, distance_bias_max: int = 512):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.bilin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dist_bias = nn.Embedding(distance_bias_max, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.dim() == 2:   # [L,D] -> [1,L,D]
            emb = emb.unsqueeze(0)
        z = self.act(self.proj(emb))                       # [B,L,H]
        zW = self.bilin(z)                                 # [B,L,H]
        logits = torch.einsum("blh,bmh->blm", zW, z) / math.sqrt(z.shape[-1])  # [B,L,L]
        B, L, _ = logits.shape
        idx = torch.arange(L, device=logits.device)
        dist = (idx[None, :] - idx[:, None]).abs().clamp_max(self.dist_bias.num_embeddings - 1)
        db = self.dist_bias(dist)[:, :, 0]                 # [L,L]
        return logits + db.unsqueeze(0)                    # [B,L,L]

# Try to import user's BilinearContactNet if available
BilinearCls = None
try:
    from rescontact.models.contact_net import BilinearContactNet as _Bilinear
    BilinearCls = _Bilinear
except Exception:
    BilinearCls = None  # we'll just use SimpleContactNet

# -----------------------
# Load config
# -----------------------
CFG_PATH = os.environ.get("RESCONTACT_CONFIG", "configs/rescontact.yaml")
CFG_FILE = Path(CFG_PATH).resolve()
CFG = yaml.safe_load(open(CFG_FILE))
# repo root if cfg is <root>/configs/rescontact.yaml
ROOT = CFG_FILE.parent.parent

DEVICE   = pick_device(CFG["project"].get("device_preference", ["cuda", "mps", "cpu"]))
CACHE    = Path(CFG["paths"]["cache_dir"]); CACHE.mkdir(parents=True, exist_ok=True)
CKPT     = Path(CFG["paths"]["ckpt_dir"]) / "model_best.pt"

ESM_NAME = CFG["model"]["esm_model"]
WANT_DIM = int(CFG["model"]["embed_dim"])
HID_DIM  = int(CFG["model"]["hidden_dim"])
DIST_MAX = int(CFG["model"]["distance_bias_max"])

# Monitoring config
MON_CFG       = CFG.get("monitoring", {}) or {}
PSI_BINS      = int(MON_CFG.get("psi_bins", 10))
PSI_WARN      = float(MON_CFG.get("psi_warn", 0.10))
PSI_ALERT     = float(MON_CFG.get("psi_alert", 0.20))
BASELINE_REL  = MON_CFG.get("baseline_path", "monitor/baseline.json")
BASELINE_PATH = (ROOT / BASELINE_REL).resolve()

# Embedder
_EMBEDDER = ESMEmbedder(ESM_NAME, cache_dir=str(CACHE), device=DEVICE)

# -----------------------
# Model selection that also probes forward-compatibility
# -----------------------
def _score_candidate(model: nn.Module, state_dict: Dict[str, torch.Tensor] | None) -> int:
    miss, unexp = [], []
    if state_dict:
        try:
            res = model.load_state_dict(state_dict, strict=False)
            miss = list(res.missing_keys)
            unexp = list(res.unexpected_keys)
        except Exception:
            # Incompatible state dict
            miss, unexp = ["_all_"], ["_all_"]
    # Forward-probe: try [L,D] or [1,L,D] to ensure no .t() misuse
    forward_ok = False
    try:
        with torch.no_grad():
            out = model(torch.randn(5, WANT_DIM, device=DEVICE))
            if isinstance(out, tuple): out = out[0]
            if out.dim() == 3: out = out.squeeze(0)
            assert out.dim() == 2
            forward_ok = True
    except Exception:
        try:
            with torch.no_grad():
                out = model(torch.randn(1, 5, WANT_DIM, device=DEVICE))
                if isinstance(out, tuple): out = out[0]
                if out.dim() == 3: out = out.squeeze(0)
                assert out.dim() == 2
                forward_ok = True
        except Exception:
            forward_ok = False
    # score = missing+unexpected + big penalty if forward fails
    return len(miss) + len(unexp) + (0 if forward_ok else 1000)

def _build_model_and_load() -> nn.Module:
    state_dict = None
    if CKPT.exists():
        blob = torch.load(CKPT, map_location=DEVICE)
        state_dict = blob.get("model") or blob.get("state_dict") or (blob if isinstance(blob, dict) else None)

    candidates: List[Tuple[str, nn.Module]] = []
    if BilinearCls is not None:
        candidates.append(("bilinear", BilinearCls(WANT_DIM, HID_DIM, DIST_MAX)))
    candidates.append(("simple", SimpleContactNet(WANT_DIM, HID_DIM, DIST_MAX)))

    best_name, best_model, best_score = None, None, float("inf")
    for name, model in candidates:
        score = _score_candidate(model, state_dict)
        if score < best_score:
            best_name, best_model, best_score = name, model, score

    model = best_model if best_model is not None else SimpleContactNet(WANT_DIM, HID_DIM, DIST_MAX)
    model = model.to(DEVICE).eval()
    # load weights again onto chosen instance (best effort)
    if state_dict:
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass
    print(f"[server] model selected: {best_name or 'simple'}  (score={best_score})")
    return model

_MODEL = _build_model_and_load()

# -----------------------
# Robust forward that avoids .t() on 3-D
# -----------------------
@torch.no_grad()
def _forward_contacts(emb_any: torch.Tensor) -> torch.Tensor:
    """
    Feed [L,D] via 2-D path first; if the user model insists on batch, retry [1,L,D].
    Always returns [L,L].
    """
    # Try 2-D
    try:
        x2 = emb_any.squeeze(0) if emb_any.dim() == 3 else emb_any  # [L,D]
        x2 = align_embed_dim(x2, WANT_DIM)
        out = _MODEL(x2)
        if isinstance(out, tuple): out = out[0]
        if out.dim() == 3: out = out.squeeze(0)
        if out.dim() != 2: raise RuntimeError(f"Model 2-D path produced shape {tuple(out.shape)}")
        return out
    except Exception:
        # Retry 3-D
        x3 = emb_any if emb_any.dim() == 3 else emb_any.unsqueeze(0)  # [1,L,D]
        x3 = align_embed_dim(x3, WANT_DIM)
        out = _MODEL(x3)
        if isinstance(out, tuple): out = out[0]
        if out.dim() == 3: out = out.squeeze(0)
        if out.dim() != 2: raise RuntimeError(f"Model 3-D path produced shape {tuple(out.shape)}")
        return out

# -----------------------
# Inference helpers
# -----------------------
@torch.no_grad()
def predict_probs_from_sequence(seq: str) -> Tuple[np.ndarray, np.ndarray]:
    seq = seq.strip().upper()
    if not seq:
        raise ValueError("Empty sequence.")
    emb_np = _EMBEDDER.embed(seq) if hasattr(_EMBEDDER, "embed") else _EMBEDDER.embed_sequence(seq)  # [L,D]
    emb_t  = torch.from_numpy(emb_np).float().to(DEVICE)  # [L,D]
    logits = _forward_contacts(emb_t)                     # [L,L]
    probs  = torch.sigmoid(logits).detach().cpu().numpy()
    return probs, emb_np

def _npz_b64_from_probs(probs: np.ndarray, threshold: float = 0.5) -> str:
    binary = (probs >= threshold).astype(np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, probs=probs, binary=binary)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -----------------------
# Minimal PSI monitor
# -----------------------
_EPS = 1e-12
def _upper_triangle(x2: np.ndarray) -> np.ndarray:
    L = x2.shape[0]
    iu = np.triu_indices(L, 1)
    return x2[iu]

def _quantile_bins(x: np.ndarray, k: int = 10) -> np.ndarray:
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0, 1.0])
    qs = np.linspace(0, 1, k + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([edges[0]-1e-6, edges[0]+1e-6])
    return edges

def _hist_props(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(len(edges)-1, 1.0/(len(edges)-1))
    h, _ = np.histogram(x, bins=edges)
    p = h.astype(np.float64)
    s = p.sum()
    return (p/s) if s > 0 else np.full_like(p, 1.0/len(p), dtype=np.float64)

def _psi_val(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), _EPS, 1.0)
    q = np.clip(np.asarray(q, float), _EPS, 1.0)
    return float(np.sum((p - q) * np.log(p / q)))

def _psi_cat(v: float) -> str:
    if v < PSI_WARN:  return "stable"
    if v < PSI_ALERT: return "slight_shift"
    if v < max(PSI_ALERT * 2.0, 0.5): return "moderate_shift"
    return "major_shift"

class _PSIMonitor:
    def __init__(self, bins: int = 10, baseline_path: Optional[Path] = None):
        self.bins = bins
        self.baseline_path = baseline_path
        self.m: Dict[str, Dict[str, Any]] = {}   # metric -> {edges, base, cur, n}
        self.counters = {"requests_total": 0, "errors_total": 0}
        self.lat_ms: List[float] = []

    def load(self):
        if not (self.baseline_path and self.baseline_path.exists()):
            return
        import json
        blob = json.loads(self.baseline_path.read_text())
        for name, cfg in blob.items():
            edges = np.array(cfg["edges"], float)
            base  = np.array(cfg["base_props"], float)
            self.m[name] = dict(edges=edges, base=base, cur=np.zeros_like(base), n=0)

    def observe(self, ctx: Dict[str, np.ndarray]):
        t0 = time.time()
        for name, st in self.m.items():
            edges = st.get("edges"); base = st.get("base")
            if edges is None or base is None: continue
            x = ctx.get(name)
            if x is None: continue
            q = _hist_props(x, edges)
            st["cur"] = st["cur"] + q
            st["n"] += 1
        self.lat_ms.append((time.time() - t0) * 1000.0)

    def snapshot(self) -> Dict[str, Any]:
        out = {"psi": {}, "counts": dict(self.counters), "latency_ms_p50_p95": None}
        if self.lat_ms:
            lat = np.array(self.lat_ms, float)
            out["latency_ms_p50_p95"] = [float(np.percentile(lat, 50)), float(np.percentile(lat, 95))]
        for name, st in self.m.items():
            if st.get("n", 0) <= 0: continue
            p = st["base"]; q = st["cur"] / max(st["n"], 1)
            val = _psi_val(p, q)
            out["psi"][name] = {"value": val, "category": _psi_cat(val), "bins": len(p)}
        return out

    def reset(self):
        for st in self.m.values():
            st["cur"][:] = 0
            st["n"] = 0
        self.lat_ms.clear()

    def prometheus(self) -> str:
        lines = []
        for k, v in self.counters.items():
            lines.append(f"# TYPE rescontact_{k} counter")
            lines.append(f"rescontact_{k} {int(v)}")
        snap = self.snapshot()
        if snap.get("latency_ms_p50_p95"):
            p50, p95 = snap["latency_ms_p50_p95"]
            lines.append("# TYPE rescontact_latency_ms gauge")
            lines.append(f'rescontact_latency_ms{{quantile="0.50"}} {p50:.3f}')
            lines.append(f'rescontact_latency_ms{{quantile="0.95"}} {p95:.3f}')
        for name, v in snap["psi"].items():
            lines.append("# TYPE rescontact_psi gauge")
            lines.append(f'rescontact_psi{{metric="{name}",category="{v["category"]}"}} {v["value"]:.6f}')
        return "\n".join(lines) + "\n"

# Init monitor and load baseline (edges + base proportions)
_MON = _PSIMonitor(bins=PSI_BINS, baseline_path=BASELINE_PATH)
_MON.load()

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
    # count request up-front (so /metrics isn't stuck at 0)
    _MON.counters["requests_total"] += 1
    try:
        if req.sequence:
            seq = req.sequence.strip().upper()
        elif req.fasta_path:
            seq = load_fasta_first(req.fasta_path)
        elif req.pdb_path:
            seq = sequence_from_pdb_quick(req.pdb_path)
        else:
            raise HTTPException(status_code=400, detail="Provide one of: sequence | fasta_path | pdb_path")

        thr = float(req.threshold)

        probs, emb_np = predict_probs_from_sequence(seq)   # [L,L], [L,D]
        L = probs.shape[0]

        # Monitor observations
        ii, jj = np.where(probs >= thr)
        pos_distance = np.abs(ii - jj).astype(float) if ii.size else np.array([], dtype=float)
        emb_norms = np.linalg.norm(emb_np, axis=1)
        msa_cov = 0.0
        if emb_np.shape[1] >= 341:
            last21 = emb_np[:, -21:]
            msa_cov = float((last21 != 0).sum()) / float(last21.size)

        _MON.observe({
            "seq_len": np.array([L], float),
            "prob_scores": _upper_triangle(probs),
            "pos_distance": pos_distance if pos_distance.size else np.array([0.0], float),
            "emb_norms": emb_norms,
            "msa_coverage": np.array([msa_cov], float),
        })

        b64 = _npz_b64_from_probs(probs, threshold=thr)
        return {"L": int(L), "threshold": thr, "npz_b64": b64}

    except HTTPException:
        _MON.counters["errors_total"] += 1
        raise
    except Exception as e:
        _MON.counters["errors_total"] += 1
        raise HTTPException(status_code=500, detail=str(e))

# ---- Visualization ----
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
            seq = req.sequence.strip().upper()
        elif req.fasta_path:
            seq = load_fasta_first(req.fasta_path)
        elif req.pdb_path:
            seq = sequence_from_pdb_quick(req.pdb_path)
        else:
            raise HTTPException(status_code=400, detail="Provide one of: sequence | fasta_path | pdb_path")

        probs, _ = predict_probs_from_sequence(seq)
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

# ---- Monitoring endpoints ----
@app.get("/psi")
def psi_snapshot():
    return _MON.snapshot()

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return _MON.prometheus()

@app.post("/admin/reset_psis")
def reset_psis():
    _MON.reset()
    return {"ok": True}

