# src/rescontact/features/embedding.py
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


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
        self.model_id = model_id
        self.cache_root = Path(cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.emb_dir = self.cache_root / "emb"
        self.emb_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        self._tok: Optional[AutoTokenizer] = None
        self._mdl: Optional[AutoModel] = None

        self.verbose = bool(int(os.environ.get("RESCONTACT_VERBOSE", "1")))
        if self.verbose:
            print(f"[rescontact/ESM] init model_id={model_id} cache={self.emb_dir} device={device}", flush=True)

    def _ensure_loaded(self):
        if self._tok is not None and self._mdl is not None:
            return
        if self.verbose:
            print(f"[rescontact/ESM] Loading {self.model_id} on {self.device} ...", flush=True)
        # Tokenizer+model (HF hub)
        # NOTE: use base AutoModel (EsmModel) – we just need hidden states
        self._tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self._mdl = AutoModel.from_pretrained(self.model_id)
        self._mdl.to(self.device).eval()
        if self.verbose:
            # infer embed dim from config if available
            try:
                d = int(self._mdl.config.hidden_size)
                print(f"[rescontact/ESM] Loaded OK (hidden_size={d})", flush=True)
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
            X = X[1 : 1 + L]

        # Always float32 (MPS prefers fp32)
        X = X.detach().to("cpu", dtype=torch.float32).numpy()
        np.savez_compressed(npz_path, h=X)
        return X

