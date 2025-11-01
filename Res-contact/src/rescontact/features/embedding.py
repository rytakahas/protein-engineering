# src/rescontact/features/embedding.py
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Hugging Face (load base ESM model WITHOUT a pooling head)
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
try:
    from transformers import EsmModel, EsmConfig  # preferred if available
    _HAS_ESM = True
except Exception:
    EsmModel = None
    EsmConfig = None
    _HAS_ESM = False


def _sha16(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()[:16]


class ESMEmbedder:
    """
    Lightweight ESM2 embedder using Hugging Face Transformers.

    - Model ID examples:
        facebook/esm2_t6_8M_UR50D        (tiny, good for laptop)
        facebook/esm2_t12_35M_UR50D      (bigger)
    - Caches per-sequence embeddings to NPZ in `cache_dir/emb/`.

    Returns numpy float32 arrays of shape [L, D].
    """

    def __init__(self, model_id: str, cache_dir: str | Path, device: torch.device):
        # Silence noisy HF warnings by default (can override via env)
        hf_logging.set_verbosity(os.environ.get("TRANSFORMERS_VERBOSITY", "error").lower())

        self.model_id = model_id
        self.cache_root = Path(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.emb_dir = self.cache_root / "emb"
        self.emb_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self._tok: Optional[AutoTokenizer] = None
        self._mdl: Optional[torch.nn.Module] = None

        self.verbose = bool(int(os.environ.get("RESCONTACT_VERBOSE", "1")))
        if self.verbose:
            print(f"[rescontact/ESM] init model_id={model_id} cache={self.emb_dir} device={device}", flush=True)

    def _ensure_loaded(self):
        if self._tok is not None and self._mdl is not None:
            return
        if self.verbose:
            print(f"[rescontact/ESM] Loading {self.model_id} on {self.device} ...", flush=True)

        # Tokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        # Model (prefer EsmModel WITHOUT a pooling layer to avoid pooler warnings)
        if _HAS_ESM:
            try:
                cfg = EsmConfig.from_pretrained(self.model_id)
                if hasattr(cfg, "add_pooling_layer"):
                    cfg.add_pooling_layer = False
                self._mdl = EsmModel.from_pretrained(self.model_id, config=cfg)
            except Exception:
                # Fallback to AutoModel if EsmModel path fails for any reason
                self._mdl = AutoModel.from_pretrained(self.model_id)
        else:
            self._mdl = AutoModel.from_pretrained(self.model_id)

        # Device + eval
        self._mdl.to(self.device).eval()

        if self.verbose:
            try:
                d = int(getattr(self._mdl.config, "hidden_size", 0))
                if d > 0:
                    print(f"[rescontact/ESM] Loaded OK (hidden_size={d})", flush=True)
                else:
                    print("[rescontact/ESM] Loaded OK", flush=True)
            except Exception:
                print("[rescontact/ESM] Loaded OK", flush=True)

    @torch.no_grad()
    def embed(self, seq: str) -> np.ndarray:
        """
        Compute per-residue hidden states [L, D] and cache to disk.
        Strips special tokens ([CLS]/[EOS]) as needed.
        """
        key = _sha16(seq)
        npz_path = self.emb_dir / f"{key}.npz"
        if npz_path.exists():
            arr = np.load(npz_path)["h"]
            return arr

        self._ensure_loaded()
        assert self._tok is not None and self._mdl is not None

        toks = self._tok(seq, return_tensors="pt", add_special_tokens=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}

        out = self._mdl(**toks).last_hidden_state  # [1, L+special, D]
        X = out[0]  # [L+special, D]

        # Strip special tokens if present
        L = len(seq)
        if X.shape[0] >= L + 2:
            X = X[1: 1 + L]

        # Always float32 (MPS prefers fp32)
        X = X.detach().to("cpu", dtype=torch.float32).numpy()
        np.savez_compressed(npz_path, h=X)
        return X
