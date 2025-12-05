
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import pandas as pd


def generate_point_mutants(
    wt_seq: str,
    ranked_positions: List[int],
    alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
    per_pos_k: int = 5,
) -> List[Tuple[int, str]]:
    muts = []
    for pos in ranked_positions[:]:
        orig = wt_seq[pos]
        count = 0
        for aa in alphabet:
            if aa == orig:
                continue
            muts.append((pos, aa))
            count += 1
            if count >= per_pos_k:
                break
    return muts


def save_mutant_csv(
    query_id: str,
    wt_seq: str,
    ranking_json: Path,
    out_csv: Path,
    per_pos_k: int = 5,
):
    data = json.loads(Path(ranking_json).read_text())
    order = data["order"]
    muts = generate_point_mutants(wt_seq, order, per_pos_k=per_pos_k)
    rows = []
    for pos, aa in muts:
        mut_seq = wt_seq[:pos] + aa + wt_seq[pos + 1 :]
        rows.append(
            {"query_id": query_id, "pos": pos, "mut_aa": aa, "wt_aa": wt_seq[pos], "mut_seq": mut_seq}
        )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
