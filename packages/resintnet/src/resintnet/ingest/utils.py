from typing import Dict, List
import pandas as pd
from Bio.Align import PairwiseAligner

def map_author_to_ref_indices(author_seq: str, ref_seq: str, author_indices_1based: List[int]) -> Dict[int,int]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    alns = aligner.align(author_seq, ref_seq)
    amap = {}
    a_blocks = alns[0].aligned[0]
    r_blocks = alns[0].aligned[1]
    for (a_start, a_end), (r_start, r_end) in zip(a_blocks, r_blocks):
        for i in range(a_end - a_start):
            amap[a_start + i + 1] = r_start + i + 1
    return {idx: amap.get(idx) for idx in author_indices_1based}

def load_table_auto(path: str) -> pd.DataFrame:
    if path.endswith(".csv"): return pd.read_csv(path)
    if path.endswith(".tsv") or path.endswith(".tab"): return pd.read_csv(path, sep="\t")
    if path.endswith(".xlsx") or path.endswith(".xls"): return pd.read_excel(path)
    return pd.read_csv(path)
