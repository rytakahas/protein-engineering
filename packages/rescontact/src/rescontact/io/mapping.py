# src/rescontact/templates/mapping.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    # pairwise2 is deprecated upstream but still shipped in Biopython 1.83
    from Bio import pairwise2
except Exception:
    pairwise2 = None


@dataclass
class MappingResult:
    """Residue index mapping + basic quality stats."""
    q_to_t: List[int]              # length = len(query_seq); -1 where unmapped
    identity: float                # fraction of identical matches over aligned columns
    coverage: float                # fraction of query residues that were mapped (q_to_t != -1)
    reason: str                    # 'from_hit_alignment' | 'naive_global' | 'empty'


def _extract_alignment_from_hit(hit: Dict) -> Optional[Tuple[str, str, int, int]]:
    """
    Try to pull aligned strings and starts from a hit dict coming from your mmseqs client.

    Supported keys (any of these):
      - 'qaln' / 'taln'  (MMseqs JSON style)
      - 'q_aln' / 't_aln' (alternative naming)
      - starts: 'qstart'/'tstart' (1-based) or 'q_start'/'t_start'
    """
    qaln = hit.get("qaln")
    taln = hit.get("taln")
    if qaln is None or taln is None:
        qaln = hit.get("q_aln")
        taln = hit.get("t_aln")
    if qaln is None or taln is None:
        return None

    qstart = hit.get("qstart", hit.get("q_start", 1))
    tstart = hit.get("tstart", hit.get("t_start", 1))
    # ensure int, 1-based in MMseqs; we’ll convert to 0-based later
    try:
        qstart = int(qstart)
        tstart = int(tstart)
    except Exception:
        qstart, tstart = 1, 1
    return str(qaln), str(taln), qstart, tstart


def _mapping_from_aligned_strings(
    qaln: str, taln: str, qstart_1b: int, tstart_1b: int, query_len: int
) -> MappingResult:
    """
    Build q->t mapping from two gapped aligned strings and 1-based starts.
    """
    qi = qstart_1b - 1  # 0-based query index
    ti = tstart_1b - 1  # 0-based template index

    q_to_t = [-1] * query_len
    aligned_cols = 0
    identical = 0

    for qc, tc in zip(qaln, taln):
        q_advance = qc != "-"
        t_advance = tc != "-"

        if q_advance and t_advance:
            # aligned residue to residue
            if 0 <= qi < query_len:
                q_to_t[qi] = ti
            aligned_cols += 1
            if qc == tc and qc != "-":
                identical += 1

        if q_advance:
            qi += 1
        if t_advance:
            ti += 1

    mapped = sum(1 for x in q_to_t if x != -1)
    coverage = mapped / float(query_len) if query_len else 0.0
    identity = (identical / float(aligned_cols)) if aligned_cols else 0.0
    return MappingResult(q_to_t=q_to_t, identity=identity, coverage=coverage, reason="from_hit_alignment")


def _naive_global_mapping(query_seq: str, template_seq: str) -> MappingResult:
    """
    Fallback: run a simple identity-scored global alignment and derive mapping.
    """
    if pairwise2 is None:
        return MappingResult(q_to_t=[-1] * len(query_seq), identity=0.0, coverage=0.0, reason="empty")

    aln = pairwise2.align.globalxx(query_seq, template_seq, one_alignment_only=True)
    if not aln:
        return MappingResult(q_to_t=[-1] * len(query_seq), identity=0.0, coverage=0.0, reason="empty")

    qaln, taln, *_ = aln[0]
    return _mapping_from_aligned_strings(qaln, taln, 1, 1, len(query_seq))


def mapping_from_hit_or_naive(
    hit: Dict,
    query_seq: str,
    template_seq: str,
    allow_naive: bool = False,
) -> MappingResult:
    """
    Preferred entry point used by build_template_priors.py.

    Order:
      1) if the hit contains aligned strings -> use them.
      2) else if allow_naive -> pairwise2 global identity alignment.
      3) else -> empty mapping.
    """
    # 1) alignment from hit, if present
    ext = _extract_alignment_from_hit(hit)
    if ext is not None:
        qaln, taln, qstart, tstart = ext
        return _mapping_from_aligned_strings(qaln, taln, qstart, tstart, len(query_seq))

    # 2) naive global alignment, if allowed
    if allow_naive:
        return _naive_global_mapping(query_seq, template_seq)

    # 3) empty
    return MappingResult(q_to_t=[-1] * len(query_seq), identity=0.0, coverage=0.0, reason="empty")


__all__ = ["MappingResult", "mapping_from_hit_or_naive"]

