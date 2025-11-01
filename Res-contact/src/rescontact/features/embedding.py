# src/rescontact/features/embedding.py
from __future__ import annotations

import hashlib
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
try:
    from transformers import EsmModel, EsmConfig
    _HAS_ESM = True
except Exception:
    EsmModel = None
    EsmConfig = None
    _HAS_ESM = False


def _sha16(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()[:16]


def _set_hf_verbosity_from_env() -> None:
    """
    Accepts TRANSFORMERS_VERBOSITY as:
      - named:  "critical|error|warning|info|debug|detail"
      - numeric: "50|40|30|20|10" (or int values)
    Falls back to ERROR if unrecognized.
    """
    val = os.environ.get("TRANSFORMERS_VERBOSITY", "error")
    level: int

    # numeric string or int
    try:
        level = int(val)
    except Exception:
        name = str(val).strip().lower()
        name_to_level = {
            "critical": logging.CRITICAL,
            "error":    logging.ERROR,
            "warning":  logging.WARNING,
            "info":     logging.INFO,
            "debug":    logging.DEBUG,
            "detail":   logging.DEBUG,  # map "detail" to DEBUG
        }
        level = name_to_level.get(name, logging.ERROR)

    hf_logging.set_verbosity(level)


class ESMEmbedder:
    """
    Lightweight ESM2 embedder using Hugging Face Transformers.

    Examples:
      - facebook/esm2_t6_8M_UR50D
      - facebook/esm2_t12_35M_UR50D

    Caches per-sequence embeddings to NPZ in `cache_dir/emb/`.
    Returns numpy float32 arrays of shape [L, D].
    """

    def __init__(self, model_id: str, cache_dir: str | Path, device: torch.device):
        # Robust logger setup
        _set_hf_verbosity_from_env()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

        # ESM uses a slow tokenizer; be explicit
        self._tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)

        # Prefer EsmModel without pooling if available
        if _HAS_ESM:
            try:
                cfg = EsmConfig.from_pretrained(self.model_id)
                if hasattr(cfg, "add_pooling_layer"):
                    cfg.add_pooling_layer = False
                self._mdl = EsmModel.from_pretrained(self.model_id, config=cfg)
            except Exception:
                self._mdl = AutoModel.from_pretrained(self.model_id)
        else:
            self._mdl = AutoModel.from_pretrained(self.model_id)

        self._mdl.to(self.device).eval()

        if self.verbose:
            d = int(getattr(self._mdl.config, "hidden_size", 0) or 0)
            msg = f"[rescontact/ESM] Loaded OK (hidden_size={d})" if d > 0 else "[rescontact/ESM] Loaded OK"
            print(msg, flush=True)

    @torch.no_grad()
    def embed(self, seq: str) -> np.ndarray:
        """
        Compute per-residue hidden states [L, D] and cache to disk.
        Strips special tokens ([CLS]/[EOS]) as needed.
        """
        key = _sha16(seq)
        npz_path = self.emb_dir / f"{key}.npz"
        if npz_path.exists():
            return np.load(npz_path)["h"]

        self._ensure_loaded()
        assert self._tok is not None and self._mdl is not None

        toks = self._tok(seq, return_tensors="pt", add_special_tokens=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}

        out = self._mdl(**toks).last_hidden_state  # [1, L+special, D]
        X = out[0]  # [L+special, D]

        L = len(seq)
        if X.shape[0] >= L + 2:  # drop BOS/EOS if present
            X = X[1:1 + L]

        X = X.detach().to("cpu", dtype=torch.float32).numpy()
        np.savez_compressed(npz_path, h=X)
        return X
