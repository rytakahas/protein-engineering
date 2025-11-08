from typing import Optional, Any
import pandas as pd
import yaml
from ..base import NormalizedMutation, finalize_mutations_frame, aa3_to_aa1, ensure_labels_int01
from ..utils import load_table_auto, map_author_to_ref_indices

def load_generic_with_mapping(path: str, mapping_yaml: str, ref_seq: Optional[str] = None) -> pd.DataFrame:
    with open(mapping_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    df = load_table_auto(path)
    prot_col = cfg.get("protein_id")
    chain_col = cfg.get("chain_id")
    res_col = cfg.get("residue_index")
    wt_col = cfg.get("wt_col")
    mut_col = cfg.get("mut_col")
    idx_mode = cfg.get("indexing","1-based")
    distal_col = cfg.get("label_distal")
    allo_col = cfg.get("label_allosteric")
    evid_src = cfg.get("evidence_source","generic_csv")
    default_chain = cfg.get("default_chain")
    default_weight = cfg.get("default_weight", 1.0)
    aa_format = cfg.get("aa_format","auto")
    author_seq_col = cfg.get("author_seq_col")

    if chain_col is None and default_chain is not None:
        df["__chain__"] = default_chain
        chain_col = "__chain__"

    def norm_aa(val: Any) -> Optional[str]:
        if val is None: return None
        s = str(val).strip()
        if not s: return None
        up = s.upper()
        if aa_format == "aa1" or len(up) == 1: return up
        if aa_format == "aa3" or len(up) == 3: return aa3_to_aa1(up)
        return up if len(up)==1 else aa3_to_aa1(up)

    if wt_col and (wt_col in df.columns): df["wt_aa1"] = df[wt_col].map(norm_aa)
    else: df["wt_aa1"] = None
    if mut_col and (mut_col in df.columns): df["mut_aa1"] = df[mut_col].map(norm_aa)
    else: df["mut_aa1"] = None

    if idx_mode == "0-based":
        df["res_idx_1based"] = df[res_col].astype(int) + 1
    elif idx_mode == "1-based":
        df["res_idx_1based"] = df[res_col].astype(int)
    elif idx_mode == "author":
        if ref_seq is None: raise ValueError("indexing=author requires ref_seq")
        if author_seq_col and author_seq_col in df.columns:
            aseq = str(df[author_seq_col].iloc[0])
        else:
            raise ValueError("author indexing requires author_seq_col in file or preprovided")
        unique_idxs = sorted(set(df[res_col].astype(int).tolist()))
        amap = map_author_to_ref_indices(aseq, ref_seq, unique_idxs)
        df["res_idx_1based"] = df[res_col].astype(int).map(lambda i: amap.get(i))
    else:
        raise ValueError(f"Unknown indexing mode: {idx_mode}")

    if distal_col and distal_col in df.columns: df["label_distal"] = df[distal_col]
    else: df["label_distal"] = None
    if allo_col and allo_col in df.columns: df["label_allosteric"] = df[allo_col]
    else: df["label_allosteric"] = None

    out = []
    for _, r in df.iterrows():
        if pd.isna(r["res_idx_1based"]): 
            continue
        out.append(NormalizedMutation(
            protein_id=str(r[prot_col]),
            chain_id=str(r[chain_col]) if chain_col and pd.notna(r[chain_col]) else None,
            res_idx_1based=int(r["res_idx_1based"]),
            wt_aa1=r.get("wt_aa1"),
            mut_aa1=r.get("mut_aa1"),
            label_distal=int(r["label_distal"]) if pd.notna(r["label_distal"]) else None,
            label_allosteric=int(r["label_allosteric"]) if pd.notna(r["label_allosteric"]) else None,
            evidence_source=evid_src,
            evidence_weight=float(default_weight) if default_weight is not None else None,
            reference=None,
            notes=None
        ))
    muts = finalize_mutations_frame(out)
    muts = ensure_labels_int01(muts, ["label_distal","label_allosteric"])
    return muts
