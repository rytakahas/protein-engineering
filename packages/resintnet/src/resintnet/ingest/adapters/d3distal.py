from typing import Optional
import pandas as pd
from ..base import NormalizedMutation, finalize_mutations_frame, aa3_to_aa1, ensure_labels_int01
from ..utils import load_table_auto, map_author_to_ref_indices

def load_d3distal(
    path: str,
    default_protein_col: str = "protein_id",
    default_chain_col: str = "chain_id",
    residue_col: str = "res_idx",
    wt_col: Optional[str] = "wt_aa",
    mut_col: Optional[str] = "mut_aa",
    label_col: str = "label_distal",
    aa_format: str = "auto",
    author_seq_col: Optional[str] = None,
    ref_seq: Optional[str] = None,
    indexing: str = "1-based",
    evidence_source: str = "d3distal",
    default_weight: Optional[float] = 1.0,
) -> pd.DataFrame:
    df = load_table_auto(path)
    if default_protein_col not in df.columns:
        for c in ["Protein","UniProt","uniprot","uniprot_id"]:
            if c in df.columns:
                df[default_protein_col] = df[c]; break
    if default_chain_col not in df.columns:
        df[default_chain_col] = "A"

    def norm_aa(x):
        if x is None: return None
        s = str(x).strip()
        if not s: return None
        up = s.upper()
        if len(up) == 1: return up
        return aa3_to_aa1(up)

    if wt_col and wt_col in df.columns: df["wt_aa1"] = df[wt_col].map(norm_aa)
    else: df["wt_aa1"] = None
    if mut_col and mut_col in df.columns: df["mut_aa1"] = df[mut_col].map(norm_aa)
    else: df["mut_aa1"] = None

    if indexing == "0-based":
        df["res_idx_1based"] = df[residue_col].astype(int) + 1
    elif indexing == "1-based":
        df["res_idx_1based"] = df[residue_col].astype(int)
    elif indexing == "author":
        if (author_seq_col is None) or (ref_seq is None):
            raise ValueError("author indexing requires author_seq_col and ref_seq")
        aseq = str(df[author_seq_col].iloc[0])
        unique_idxs = sorted(set(df[residue_col].astype(int).tolist()))
        amap = map_author_to_ref_indices(aseq, ref_seq, unique_idxs)
        df["res_idx_1based"] = df[residue_col].astype(int).map(lambda i: amap.get(i))
    else:
        raise ValueError(f"Unknown indexing mode: {indexing}")

    if label_col in df.columns: df["label_distal"] = df[label_col]
    else: df["label_distal"] = 1

    df["label_allosteric"] = None
    df["evidence_source"] = evidence_source
    df["evidence_weight"] = default_weight
    df["reference"] = None
    df["notes"] = None

    out = []
    for _, r in df.iterrows():
        if pd.isna(r["res_idx_1based"]): 
            continue
        out.append(NormalizedMutation(
            protein_id=str(r[default_protein_col]),
            chain_id=str(r[default_chain_col]) if pd.notna(r[default_chain_col]) else None,
            res_idx_1based=int(r["res_idx_1based"]),
            wt_aa1=r.get("wt_aa1"),
            mut_aa1=r.get("mut_aa1"),
            label_distal=int(r["label_distal"]) if pd.notna(r["label_distal"]) else None,
            label_allosteric=None,
            evidence_source=evidence_source,
            evidence_weight=default_weight,
            reference=None,
            notes=None
        ))
    muts = finalize_mutations_frame(out)
    muts = ensure_labels_int01(muts, ["label_distal","label_allosteric"])
    return muts
