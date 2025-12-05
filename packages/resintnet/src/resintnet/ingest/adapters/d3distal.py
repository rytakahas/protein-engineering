# src/resintnet/ingest/adapters/d3distal.py

from pathlib import Path
import pandas as pd

def load_d3distal(path):
    """
    Load a D3Distal-style file into a standard DMS DataFrame with columns:
      uniprot (optional), pos (int), wt (1-letter), mut (1-letter), score (float)
    Accepts CSV or TSV and normalizes common header variants.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"D3Distal file not found: {p}")

    sep = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(p, sep=sep)

    # normalize header names
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"uniprot", "accession", "acc", "protein", "protein_id"}:
            rename[c] = "uniprot"
        elif cl in {"position", "pos", "i", "residue_index", "res_index"}:
            rename[c] = "pos"
        elif cl in {"wt", "wildtype", "aa_wt", "ref"}:
            rename[c] = "wt"
        elif cl in {"mut", "aa_mut", "variant", "alt"}:
            rename[c] = "mut"
        elif cl in {"score", "fitness", "effect", "delta_delta_g", "ddg"}:
            rename[c] = "score"

    df = df.rename(columns=rename)

    # required fields
    missing = [c for c in ["pos", "wt", "mut", "score"] if c not in df.columns]
    if missing:
        raise ValueError(f"D3Distal file missing required columns: {missing}. Got: {list(df.columns)}")

    # types & cleanup
    df["pos"] = df["pos"].astype(int)
    df["wt"] = df["wt"].astype(str).str.upper().str[0]
    df["mut"] = df["mut"].astype(str).str.upper().str[0]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)

    if "uniprot" not in df.columns:
        df["uniprot"] = None

    return df[["uniprot", "pos", "wt", "mut", "score"]]

