
import pandas as pd
from ..base import Variant, DMSSet

def load_dms_csv(path: str, uniprot: str) -> DMSSet:
    df = pd.read_csv(path)
    req = {"pos","wt","mut","score"}
    lc = {c.lower(): c for c in df.columns}
    if not req.issubset(set(lc.keys())):
        raise ValueError(f"DMS CSV must contain columns {req}")
    pos = df[lc["pos"]].astype(int).tolist()
    wt  = df[lc["wt"]].astype(str).str.upper().tolist()
    mut = df[lc["mut"]].astype(str).str.upper().tolist()
    sco = pd.to_numeric(df[lc["score"]], errors="coerce").fillna(0.0).tolist()
    vars = [Variant(p, w, m, s) for p,w,m,s in zip(pos, wt, mut, sco)]
    return DMSSet(uniprot=uniprot, variants=vars)
