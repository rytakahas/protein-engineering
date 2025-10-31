# tests/test_msa_providers_mock.py
import pytest
from rescontact.features.msa import MSAProvider

def test_msa_fallbacks(tmp_path):
    cfg = {
        "local_glob": None,
        "jackhmmer": {"enabled": False},
        "blastp": {"enabled": False},
    }
    p = MSAProvider(cfg=cfg, cache_dir=tmp_path)
    out = p.get("id1", "ACDE")

    # Accept either legacy None or current (None, None) tuple
    assert (out is None) or (isinstance(out, tuple) and out == (None, None))

