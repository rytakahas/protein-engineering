import os
import glob
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Remote BLAST (optional); keep imports local and guarded
try:
    from Bio.Blast import NCBIWWW, NCBIXML  # noqa: F401
    _BIO_BLAST_OK = True
except Exception:
    _BIO_BLAST_OK = False


_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_AA2IDX = {a: i for i, a in enumerate(_AA20)}


def _sha16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_alignment_as_rows(path: str, query_seq: str, max_seqs: Optional[int] = None) -> List[str]:
    """
    Read A3M/FASTA-like alignment and return rows aligned to the query (length L).
    Very lightweight parser:
      - ignores header lines (starting with '>')
      - keeps dashes '-'
      - strips whitespace
      - converts to uppercase
      - ignores rows with different length from the query
    """
    rows: List[str] = []
    L = len(query_seq)
    with open(path, "r") as f:
        cur = []
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    s = "".join(cur).strip().replace(" ", "")
                    s = s.upper()
                    if len(s) == L:
                        rows.append(s)
                    cur = []
                continue
            cur.append(line.strip())
        if cur:
            s = "".join(cur).strip().replace(" ", "")
            s = s.upper()
            if len(s) == L:
                rows.append(s)

    # Ensure first row is the query itself if missing
    if not rows or rows[0] != query_seq.upper():
        # Build query row as exact sequence (no gaps)
        rows = [query_seq.upper()] + rows

    if max_seqs is not None and len(rows) > max_seqs:
        rows = rows[:max_seqs]
    return rows


def msa_1d_features(rows: List[str]) -> np.ndarray:
    """
    Simple PSSM-like features: per-position amino-acid frequencies (20) + entropy (1).
    Returns [L, 21] float32.
    """
    assert len(rows) > 0
    L = len(rows[0])
    X = np.zeros((L, 21), dtype=np.float32)

    # Count frequencies over rows (skip gaps '-')
    for j in range(L):
        col = [r[j] for r in rows if j < len(r)]
        counts = np.zeros(20, dtype=np.float32)
        n = 0
        for ch in col:
            if ch == "-" or ch == ".":
                continue
            a = ch.upper()
            if a in _AA2IDX:
                counts[_AA2IDX[a]] += 1.0
                n += 1
        if n > 0:
            freqs = counts / float(n)
        else:
            freqs = counts  # zeros
        # Entropy (base e); avoid log(0)
        eps = 1e-8
        ent = -np.sum(freqs * np.log(freqs + eps), dtype=np.float32)
        X[j, :20] = freqs
        X[j, 20] = ent
    return X


class MSAProvider:
    """
    Provider that can:
      1) Use local A3M files (glob pattern)
      2) Use cached A3M under cache_dir
      3) Optionally call remote BLASTP (NCBI qblast) and convert XML→A3M,
         honoring timeouts and caching results on disk.

    Config (example):
      {
        "provider_order": ["local", "blastp"],
        "local_glob": "data/msa/**/*.a3m",
        "blastp": {
          "enabled": true,
          "db": "swissprot",
          "hitlist_size": 50,
          "expect": 1e-3,
          "timeout_s": 120,
          "max_seqs": 64
        }
      }
    """

    def __init__(self, cfg: Dict, cache_dir: Optional[str] = None):
        self.cfg = cfg or {}
        self.provider_order: List[str] = list(self.cfg.get("provider_order", ["local"]))
        self.local_glob: Optional[str] = self.cfg.get("local_glob")
        self.remote_cfg: Dict = self.cfg.get("blastp", {}) or {}

        # Where to write cached A3M / XML artifacts
        root_cache = cache_dir or ".cache/rescontact"
        self.cache_a3m_dir = Path(root_cache) / "msa" / "a3m"
        self.cache_xml_dir = Path(root_cache) / "msa" / "xml"
        _ensure_dir(self.cache_a3m_dir)
        _ensure_dir(self.cache_xml_dir)

    # ---------- public ----------

    def get(self, sequence_id: str, seq: str) -> Optional[str]:
        """
        Return path to A3M aligned file for this sequence (or None).
        Tries providers in order. Caches outputs in cache_dir.
        """
        # 0) Already cached A3M for this sequence?
        a3m_cached = self._cached_a3m(sequence_id)
        if a3m_cached:
            return a3m_cached

        # 1) Try providers in order
        for prov in self.provider_order:
            if prov == "local":
                p = self._from_local(sequence_id)
                if p:
                    return p
            elif prov == "blastp":
                p = self._from_remote_blast(sequence_id, seq)
                if p:
                    return p
            # (more providers in the future: jackhmmer REST, etc.)

        # Nothing found
        return None

    # ---------- local ----------

    def _from_local(self, sequence_id: str) -> Optional[str]:
        if not self.local_glob:
            return None
        # naive strategy: if any a3m is present, just return the first one;
        # or prefer a file that includes the sequence_id in the name
        files = sorted(glob.glob(self.local_glob, recursive=True))
        if not files:
            return None
        # Match id first
        for f in files:
            if sequence_id in os.path.basename(f):
                return f
        return files[0]

    # ---------- cache ----------

    def _cached_a3m(self, sequence_id: str) -> Optional[str]:
        p = self.cache_a3m_dir / f"{sequence_id}.a3m"
        return str(p) if p.exists() else None

    def _write_a3m(self, sequence_id: str, rows: List[str]) -> str:
        p = self.cache_a3m_dir / f"{sequence_id}.a3m"
        with open(p, "w") as f:
            # write as FASTA/A3M-like; first row is query
            f.write(f">{sequence_id}\n")
            f.write(rows[0] + "\n")
            for i, row in enumerate(rows[1:], start=1):
                f.write(f">{sequence_id}_hit{i}\n")
                f.write(row + "\n")
        return str(p)

    # ---------- remote BLAST ----------

    def _from_remote_blast(self, sequence_id: str, seq: str) -> Optional[str]:
        # Guard: remote disabled or Biopython not available
        if not self.remote_cfg.get("enabled", False):
            return None
        if not _BIO_BLAST_OK:
            return None

        # Respect env switch if provided
        allow_remote_env = os.environ.get("ALLOW_REMOTE_MSA", "").lower()
        if allow_remote_env in {"0", "false", "no"}:
            return None

        # Cache key is sha of seq to deduplicate
        tag = _sha16(seq)
        xml_path = self.cache_xml_dir / f"{sequence_id}-{tag}.blastp.xml"
        a3m_path = self.cache_a3m_dir / f"{sequence_id}.a3m"
        if a3m_path.exists():
            return str(a3m_path)

        # If an XML already exists (from prior run), try to convert it
        if xml_path.exists():
            rows = _blast_xml_to_rows(xml_path, seq, self.remote_cfg.get("max_seqs", 64))
            if rows:
                return self._write_a3m(sequence_id, rows)

        # Otherwise, query NCBI qblast (rate-limited; be polite)
        db = str(self.remote_cfg.get("db", "swissprot"))
        hitlist_size = int(self.remote_cfg.get("hitlist_size", 50))
        expect = float(self.remote_cfg.get("expect", 1e-3))
        timeout_s = int(self.remote_cfg.get("timeout_s", 120))

        try:
            # NCBIWWW.qblast has no timeout arg; wrap in a coarse timeout
            t0 = time.time()
            handle = NCBIWWW.qblast(
                program="blastp",
                database=db,
                sequence=seq,
                hitlist_size=hitlist_size,
                expect=expect,
                format_type="XML",
            )
            xml = handle.read()
            with open(xml_path, "w") as f:
                f.write(xml)
            if (time.time() - t0) > timeout_s:
                # Timed out from our perspective: do not block training
                return None
            rows = _blast_xml_text_to_rows(xml, seq, self.remote_cfg.get("max_seqs", 64))
            if rows:
                return self._write_a3m(sequence_id, rows)
        except Exception:
            # Network/server/parse failure → return None (caller will pad zeros)
            return None
        return None


# ---------- BLAST XML → alignment rows ----------

def _blast_xml_to_rows(xml_path: Path, query_seq: str, max_seqs: int) -> List[str]:
    try:
        with open(xml_path, "r") as f:
            xml = f.read()
        return _blast_xml_text_to_rows(xml, query_seq, max_seqs)
    except Exception:
        return []


def _blast_xml_text_to_rows(xml_text: str, query_seq: str, max_seqs: int) -> List[str]:
    """
    Convert BLAST XML (single query) to a set of query-aligned rows of length L.
    We use the top HSP per hit. Very lightweight and conservative.
    """
    try:
        from Bio.Blast import NCBIXML
    except Exception:
        return []

    L = len(query_seq)
    rows: List[str] = []
    try:
        rec = next(NCBIXML.parse(iter([xml_text])))
    except Exception:
        # Fallback: write query only
        return [query_seq.upper()]

    # First row: the query (exact)
    rows.append(query_seq.upper())

    # Then subject rows
    for aln in rec.alignments:
        if not aln.hsps:
            continue
        hsp = aln.hsps[0]  # top HSP

        # hsp.query and hsp.sbjct are aligned segments with '-' gaps
        q = str(hsp.query)   # may contain '-'
        s = str(hsp.sbjct)   # may contain '-'

        # Build a query-length row, fill residues aligned to query positions
        row = ["-"] * L
        qpos = 0  # position in the full query sequence (0..L-1)
        # We need to map the aligned subsegment back to positions in the full query.
        # Use hsp.query_start (1-based in BLAST) to place q characters.
        qpos = max(0, int(hsp.query_start) - 1)

        for qc, sc in zip(q, s):
            if qc == "-":
                # insertion w.r.t query → skip increasing qpos
                continue
            # qc is a query residue; we are at this query position
            if 0 <= qpos < L:
                if sc != "-":
                    row[qpos] = sc.upper()
            qpos += 1

        rows.append("".join(row))

        if len(rows) >= max(1, int(max_seqs)):
            break

    return rows if rows else [query_seq.upper()]

