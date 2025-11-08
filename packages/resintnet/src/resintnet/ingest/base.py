from dataclasses import dataclass, asdict
from typing import Optional, Iterable, List
import pandas as pd

AA3_TO_AA1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLU":"E","GLN":"Q","GLY":"G","HIS":"H",
    "ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W",
    "TYR":"Y","VAL":"V","SEC":"U","PYL":"O"
}
AA1 = set("ACDEFGHIKLMNPQRSTVWY")
AA1_WITH_RARE = AA1 | set("UO")

@dataclass
class NormalizedMutation:
    protein_id: str
    chain_id: Optional[str]
    res_idx_1based: int
    wt_aa1: Optional[str]
    mut_aa1: Optional[str]
    label_distal: Optional[int]
    label_allosteric: Optional[int]
    evidence_source: str
    evidence_weight: Optional[float]
    reference: Optional[str]
    notes: Optional[str]

@dataclass
class NormalizedResidue:
    protein_id: str
    chain_id: Optional[str]
    res_idx_1based: int
    label_distal: Optional[int]
    label_allosteric: Optional[int]
    evidence_source: str
    evidence_weight: Optional[float]
    reference: Optional[str]
    notes: Optional[str]

REQUIRED_MUT_COLS = [
    "protein_id","chain_id","res_idx_1based","wt_aa1","mut_aa1",
    "label_distal","label_allosteric","evidence_source",
    "evidence_weight","reference","notes"
]
REQUIRED_RES_COLS = [
    "protein_id","chain_id","res_idx_1based",
    "label_distal","label_allosteric","evidence_source",
    "evidence_weight","reference","notes"
]

def aa3_to_aa1(x: Optional[str]) -> Optional[str]:
    if x is None: return None
    x = x.strip().upper()
    if len(x) == 1:
        return x if x in AA1_WITH_RARE else None
    return AA3_TO_AA1.get(x)

def ensure_labels_int01(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: 1 if str(v).strip() in {"1","true","True","yes","YES"} else (0 if str(v).strip() in {"0","false","False","no","NO"} else None))
        else:
            df[c] = None
    return df

def finalize_mutations_frame(rows: Iterable[NormalizedMutation]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    for c in REQUIRED_MUT_COLS:
        if c not in df.columns: df[c] = None
    df = df[REQUIRED_MUT_COLS]
    return df

def mutations_to_residues(muts: pd.DataFrame, agg: str = "any") -> pd.DataFrame:
    def agg01(series):
        s = [v for v in series if pd.notna(v)]
        if not s: return None
        if agg == "any":
            return 1 if any(int(v)==1 for v in s) else 0
        if agg == "max":
            return max(int(v) for v in s)
        if agg == "mean":
            return float(sum(int(v) for v in s))/len(s)
        return 1 if any(int(v)==1 for v in s) else 0
    group = muts.groupby(["protein_id","chain_id","res_idx_1based"], as_index=False).agg({
        "label_distal": agg01,
        "label_allosteric": agg01,
        "evidence_weight": "mean",
        "evidence_source": lambda s: ";".join(sorted(set([str(x) for x in s if pd.notna(x)]))),
        "reference": lambda s: ";".join(sorted(set([str(x) for x in s if pd.notna(x)]))),
        "notes": lambda s: None
    })
    for c in REQUIRED_RES_COLS:
        if c not in group.columns: group[c] = None
    return group[REQUIRED_RES_COLS]
