from rescontact.features.msa import MSAProvider

def test_msa_fallbacks():
    p = MSAProvider({"local_glob": None, "jackhmmer": {"enabled": False}, "blastp": {"enabled": False}})
    out = p.get("id1", "ACDE")
    assert out is None  # gracefully skipped

