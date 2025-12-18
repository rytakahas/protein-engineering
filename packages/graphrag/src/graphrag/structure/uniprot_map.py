from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

def _try_biopython_alignment():
    try:
        from Bio.Align import PairwiseAligner  # type: ignore
        return PairwiseAligner
    except Exception:
        return None

@dataclass(frozen=True)
class UniProtMap:
    """Mapping from PDB residue order index -> UniProt position (1-based)."""
    chain: str
    uniprot_id: str
    pdb_index_to_uniprot_pos: Dict[int, int]
    score: float
    method: str

def map_chain_to_uniprot_positions(
    *,
    chain: str,
    chain_sequence: str,
    uniprot_id: str,
    uniprot_sequence: str,
) -> Optional[UniProtMap]:
    """
    Align chain_sequence to uniprot_sequence and return position mapping.

    Requires Biopython alignment. If Biopython is unavailable, returns None.
    Caller should fallback (e.g., to sequential positions).
    """
    PairwiseAligner = _try_biopython_alignment()
    if PairwiseAligner is None:
        return None

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.2

    alns = aligner.align(uniprot_sequence, chain_sequence)
    if not alns:
        return None
    aln = alns[0]
    score = float(aln.score)

    a = uniprot_sequence
    b = chain_sequence
    a_blocks, b_blocks = aln.aligned

    i_a = 0
    i_b = 0
    out_a: List[str] = []
    out_b: List[str] = []

    for (a_start, a_end), (b_start, b_end) in zip(a_blocks, b_blocks):
        while i_a < a_start and i_b < b_start:
            out_a.append(a[i_a]); out_b.append(b[i_b])
            i_a += 1; i_b += 1
        while i_a < a_start:
            out_a.append(a[i_a]); out_b.append("-")
            i_a += 1
        while i_b < b_start:
            out_a.append("-"); out_b.append(b[i_b])
            i_b += 1
        while i_a < a_end and i_b < b_end:
            out_a.append(a[i_a]); out_b.append(b[i_b])
            i_a += 1; i_b += 1

    while i_a < len(a) and i_b < len(b):
        out_a.append(a[i_a]); out_b.append(b[i_b])
        i_a += 1; i_b += 1
    while i_a < len(a):
        out_a.append(a[i_a]); out_b.append("-")
        i_a += 1
    while i_b < len(b):
        out_a.append("-"); out_b.append(b[i_b])
        i_b += 1

    gA = "".join(out_a)
    gB = "".join(out_b)

    pdb_index_to_uniprot: Dict[int, int] = {}
    pos_u = 0
    pos_c = 0

    for ch_u, ch_c in zip(gA, gB):
        if ch_u != "-":
            pos_u += 1
        if ch_c != "-":
            pos_c += 1
        if ch_u != "-" and ch_c != "-":
            pdb_index_to_uniprot[pos_c - 1] = pos_u

    return UniProtMap(
        chain=chain,
        uniprot_id=uniprot_id,
        pdb_index_to_uniprot_pos=pdb_index_to_uniprot,
        score=score,
        method="biopython_pairwise_global",
    )
