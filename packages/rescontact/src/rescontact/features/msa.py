# src/rescontact/features/msa.py
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# -----------------------------
# Small logging utility
# -----------------------------
def _vlevel() -> int:
    # 0 = silent, 1 = normal, 2 = verbose
    try:
        return int(os.getenv("RESCONTACT_VERBOSE", "1"))
    except Exception:
        return 1

def _log(tag: str, msg: str, level: int = 1):
    if _vlevel() >= level:
        print(f"[{tag}] {msg}")


# -----------------------------
# Alignment utilities
# -----------------------------
_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = set(_AA)
_GAP_CHARS = set("-._")  # treat ., _ as gaps when normalizing

def read_alignment_as_rows(path: str, query_seq: str, max_seqs: int = 64) -> List[str]:
    """
    Read an alignment file and return a list of equal-length rows (strings).
    Accepts:
      - A3M/A2M ('.' gaps) or aligned FASTA (with '-' gaps)
      - Simple "FASTA-like" aligned outputs we write locally

    We normalize gaps to '-' and upper-case letters.
    """
    p = Path(path)
    if not p.exists():
        return []

    rows: List[str] = []
    cur: List[str] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if cur:
                    rows.append("".join(cur))
                    cur = []
                continue
            cur.append(s)
        if cur:
            rows.append("".join(cur))

    # Normalize gaps & casing
    rows = [r.replace(".", "-").replace("_", "-").upper() for r in rows if r]
    if not rows:
        return []

    # Enforce equal length
    L0 = len(rows[0])
    rows = [r for r in rows if len(r) == L0]
    if not rows:
        return []

    # Keep at most max_seqs rows
    rows = rows[:max_seqs]
    return rows


def msa_1d_features(rows: List[str]) -> "np.ndarray":
    """
    Turn an alignment (list of equal-length strings) into per-position 21-d features:
      20 amino acids + 1 gap.
    We use simple frequency (counts normalized by number of rows).
    Returns float32 array [L, 21].
    """
    import numpy as np

    if not rows:
        return np.zeros((0, 21), dtype=np.float32)

    L = len(rows[0])
    N = len(rows)
    mat = np.zeros((L, 21), dtype=np.float32)  # 20 aa + gap

    aa_index = {a: i for i, a in enumerate(_AA)}
    GAP_IDX = 20

    for row in rows:
        # defensive: assume equal length enforced by reader
        for i, ch in enumerate(row):
            if ch in _AA_SET:
                mat[i, aa_index[ch]] += 1.0
            else:
                # everything else (including '-') counts as gap
                mat[i, GAP_IDX] += 1.0

    # normalize by number of rows
    if N > 0:
        mat /= float(N)
    return mat.astype(np.float32)


# -----------------------------
# Remote provider: Jackhmmer (EMBL-EBI HMMER)
# -----------------------------
class JackhmmerRemote:
    """
    Minimal client for EMBL-EBI HMMER jackhmmer REST API.
    API endpoints:
      POST   https://www.ebi.ac.uk/Tools/hmmer/run/jackhmmer
      GET    https://www.ebi.ac.uk/Tools/hmmer/status/{jobid}
      GET    https://www.ebi.ac.uk/Tools/hmmer/results/{jobid}/aln-fasta
      GET    https://www.ebi.ac.uk/Tools/hmmer/results/{jobid}/aln-a2m
    """
    def __init__(self, cfg: dict, cache_dir: str):
        j = (cfg or {}).get("jackhmmer_remote", {}) or {}
        self.enabled      = bool(j.get("enabled", True))
        self.db           = j.get("db", "uniprotrefprot")   # or "uniprotkb"
        self.email        = j.get("email")                  # optional but recommended
        self.timeout_s    = int(j.get("timeout_s", 300))
        self.poll_every_s = int(j.get("poll_every_s", 5))
        self.max_seqs     = int(j.get("max_seqs", (cfg or {}).get("max_seqs", 64)))
        self.base         = "https://www.ebi.ac.uk/Tools/hmmer"
        self.cache_dir    = Path(cache_dir) / "msa_remote"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _submit(self, seq: str) -> Optional[str]:
        fasta = f">query\n{seq}\n"
        data = {"seq": fasta, "db": self.db}
        if self.email:
            data["email"] = self.email
        headers = {
            "User-Agent": "rescontact/0.1 (+msaclient)",
            "Accept": "text/plain",
        }
        url = f"{self.base}/run/jackhmmer"
        r = requests.post(url, data=data, headers=headers, timeout=30)
        if r.status_code == 405:
            _log("msa/jack", "405 Not Allowed from /run/jackhmmer — check URL and proxies", 1)
            return None
        r.raise_for_status()
        job_id = r.text.strip()
        if not job_id:
            _log("msa/jack", "empty job_id from jackhmmer", 1)
            return None
        _log("msa/jack", f"submitted job_id={job_id}", 2)
        return job_id

    def _status(self, job_id: str) -> str:
        r = requests.get(f"{self.base}/status/{job_id}", timeout=20)
        r.raise_for_status()
        return r.text.strip()

    def _fetch_alignment_text(self, job_id: str) -> Optional[str]:
        # try aligned FASTA first
        r = requests.get(f"{self.base}/results/{job_id}/aln-fasta",
                         timeout=60, headers={"Accept": "text/plain"})
        if r.status_code == 200 and r.text.strip():
            return r.text
        # fallback to A2M
        r = requests.get(f"{self.base}/results/{job_id}/aln-a2m",
                         timeout=60, headers={"Accept": "text/plain"})
        if r.status_code == 200 and r.text.strip():
            return r.text
        return None

    def run(self, sequence_id: str, seq: str) -> Optional[str]:
        if not self.enabled:
            return None

        job_id = self._submit(seq)
        if not job_id:
            return None

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                st = self._status(job_id)
            except Exception as e:
                _log("msa/jack", f"status error: {e}", 1)
                return None

            if st.upper() in ("FINISHED", "DONE", "SUCCESS", "COMPLETE"):
                _log("msa/jack", f"status={st}", 2)
                break
            if st.upper() in ("RUNNING", "PENDING", "QUEUED"):
                time.sleep(self.poll_every_s)
                continue

            _log("msa/jack", f"unexpected status={st}", 1)
            return None
        else:
            _log("msa/jack", "timeout waiting for jackhmmer job", 1)
            return None

        try:
            txt = self._fetch_alignment_text(job_id)
        except Exception as e:
            _log("msa/jack", f"fetch error: {e}", 1)
            return None

        if not txt:
            _log("msa/jack", "no alignment text fetched", 1)
            return None

        out = self.cache_dir / f"{sequence_id}.fasta"
        out.write_text(txt)
        _log("msa/jack", f"cached alignment → {out}", 2)
        return str(out)


# -----------------------------
# Remote provider: BLASTP (NCBI)
# -----------------------------
class BlastPRemote:
    """
    Minimal client for NCBI BLAST URLAPI.

    Flow:
      1) PUT (CMD=Put, PROGRAM=blastp, DATABASE=..., QUERY=FASTA) -> parse RID
      2) Poll (CMD=Get, FORMAT_OBJECT=SearchInfo) until Status=READY
      3) GET (CMD=Get, FORMAT_TYPE=Text, ALIGNMENTS=k) -> parse pairwise alignments,
         build aligned subject rows (same length as aligned query), write aligned FASTA.

    Notes:
      - Uses API key from env NCBI_API_KEY if present (param API_KEY)
      - We keep it simple & defensive; returns None on any error.
    """
    def __init__(self, cfg: dict, cache_dir: str):
        b = (cfg or {}).get("blastp", {}) or {}
        self.enabled      = bool(b.get("enabled", True))
        self.db           = b.get("db", "swissprot")  # or "nr"
        self.hitlist_size = int(b.get("hitlist_size", 25))
        self.expect       = str(b.get("expect", "1e-3"))
        self.timeout_s    = int(b.get("timeout_s", 120))
        self.max_seqs     = int(b.get("max_seqs", (cfg or {}).get("max_seqs", 64)))
        self.base         = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
        self.api_key      = os.getenv("NCBI_API_KEY")  # optional
        self.cache_dir    = Path(cache_dir) / "msa_remote"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- helpers ----
    def _put(self, fasta: str) -> Optional[str]:
        data = {
            "CMD": "Put",
            "PROGRAM": "blastp",
            "DATABASE": self.db,
            "QUERY": fasta,
            "HITLIST_SIZE": str(self.hitlist_size),
            "EXPECT": str(self.expect),
            "FORMAT_TYPE": "Text",  # later we'll fetch Text with CMD=Get
        }
        if self.api_key:
            data["API_KEY"] = self.api_key

        r = requests.post(self.base, data=data, timeout=30)
        r.raise_for_status()
        text = r.text
        m = re.search(r"RID =\s*([A-Z0-9\-]+)", text)
        if not m:
            _log("msa/blast", "RID not found in PUT response", 1)
            return None
        rid = m.group(1)
        _log("msa/blast", f"RID={rid}", 2)
        return rid

    def _ready(self, rid: str) -> bool:
        params = {"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"}
        if self.api_key:
            params["API_KEY"] = self.api_key
        r = requests.get(self.base, params=params, timeout=20)
        r.raise_for_status()
        t = r.text
        if "Status=READY" in t:
            return True
        if "Status=FAILED" in t or "Status=UNKNOWN" in t:
            _log("msa/blast", f"poll status shows failure/unknown: {t.strip()[:120]}", 1)
            return False
        return False

    def _fetch_text(self, rid: str) -> Optional[str]:
        params = {
            "CMD": "Get",
            "RID": rid,
            "FORMAT_TYPE": "Text",
            "ALIGNMENTS": str(self.max_seqs),
            "DESCRIPTIONS": "0",  # skip long headers
        }
        if self.api_key:
            params["API_KEY"] = self.api_key
        r = requests.get(self.base, params=params, timeout=60)
        r.raise_for_status()
        return r.text

    @staticmethod
    def _parse_pairwise_to_aligned_rows(txt: str, max_rows: int) -> List[str]:
        """
        Crude parser for BLAST Text alignments.

        Builds aligned subject rows concatenating blocks. Keeps at most `max_rows`.
        """
        rows: List[str] = []
        # Split by hits: look for lines starting with '>' to identify new subject
        blocks = re.split(r"\n>", txt)
        for b in blocks[1:]:
            # Gather all query/subject segments
            q_aln = []
            s_aln = []
            for line in b.splitlines():
                # Typical lines:
                # Query  1   MAA---VQLL...   60
                # Sbjct  5   MPAKG-VQLI...   64
                if line.startswith("Query "):
                    parts = line.split()
                    if len(parts) >= 3:
                        q_aln.append(parts[2])
                elif line.startswith("Sbjct "):
                    parts = line.split()
                    if len(parts) >= 3:
                        s_aln.append(parts[2])
            if not q_aln or not s_aln:
                continue
            q = "".join(q_aln).upper()
            s = "".join(s_aln).upper()
            # normalize gaps: keep '-' as gap, map '.' '_' to '-'
            q = q.replace(".", "-").replace("_", "-")
            s = s.replace(".", "-").replace("_", "-")
            if len(q) != len(s):
                # Alignments should match length; if not, skip this hit
                continue
            # Keep subject row; we don't strictly need to include the query row
            rows.append(s)
            if len(rows) >= max_rows:
                break
        return rows

    def run(self, sequence_id: str, seq: str) -> Optional[str]:
        if not self.enabled:
            return None

        fasta = f">query\n{seq}\n"
        try:
            rid = self._put(fasta)
        except Exception as e:
            _log("msa/blast", f"PUT error: {e}", 1)
            return None
        if not rid:
            return None

        deadline = time.time() + self.timeout_s
        ready = False
        while time.time() < deadline:
            try:
                if self._ready(rid):
                    ready = True
                    break
                time.sleep(2.5)
            except Exception as e:
                _log("msa/blast", f"poll error: {e}", 1)
                return None
        if not ready:
            _log("msa/blast", "timeout waiting for BLAST job", 1)
            return None

        try:
            text = self._fetch_text(rid)
        except Exception as e:
            _log("msa/blast", f"fetch error: {e}", 1)
            return None
        if not text:
            return None

        rows = self._parse_pairwise_to_aligned_rows(text, self.max_seqs)
        if not rows:
            _log("msa/blast", "no aligned subject rows parsed", 1)
            return None

        # Build aligned FASTA with just subjects; the reader builds features from them.
        # (Optionally we could include aligned Query as first row.)
        out = Path(self.cache_dir) / f"{sequence_id}.fasta"
        with out.open("w") as f:
            for i, r in enumerate(rows, 1):
                f.write(f">hit{i}\n{r}\n")
        _log("msa/blast", f"cached alignment → {out}", 2)
        return str(out)


# -----------------------------
# Provider Cascade
# -----------------------------
class MSAProvider:
    """
    Orchestrates MSA sources in order:
      - "local"            → search for local A3M/A2M/FASTA by glob
      - "jackhmmer_remote" → EMBL-EBI HMMER jackhmmer REST
      - "blastp"           → NCBI BLASTP URLAPI

    get(sequence_id, seq) → (path_or_none, provider_name_or_none)
    """
    def __init__(self, cfg: dict | None, cache_dir: str):
        self.cfg = cfg or {}
        order = self.cfg.get("provider_order") or []
        self.order: List[str] = list(order)
        self.max_seqs = int(self.cfg.get("max_seqs", 64))
        self.local_glob = self.cfg.get("local_glob")

        self.jack = JackhmmerRemote(self.cfg, cache_dir) if "jackhmmer_remote" in self.order else None
        self.blast = BlastPRemote(self.cfg, cache_dir) if "blastp" in self.order else None

    def _find_local(self, sequence_id: str) -> Optional[str]:
        if not self.local_glob:
            return None
        # Very simple: use the first file that contains the sequence_id stem
        # or the very first match if nothing matches exactly.
        from glob import glob

        matches = glob(self.local_glob, recursive=True)
        if not matches:
            return None
        # heuristic: prefer filename containing the id
        for m in matches:
            if sequence_id in Path(m).name:
                return m
        # otherwise, return the first
        return matches[0]

    def get(self, sequence_id: str, seq: str) -> Tuple[Optional[str], Optional[str]]:
        # 1) Local A3M/FASTA
        if "local" in self.order:
            p = self._find_local(sequence_id)
            if p:
                return p, "local"

        # 2) Jackhmmer remote
        if "jackhmmer_remote" in self.order and self.jack is not None and self.jack.enabled:
            p = self.jack.run(sequence_id=sequence_id, seq=seq)
            if p:
                return p, "jackhmmer_remote"

        # 3) BLASTP remote
        if "blastp" in self.order and self.blast is not None and self.blast.enabled:
            p = self.blast.run(sequence_id=sequence_id, seq=seq)
            if p:
                return p, "blastp"

        return None, None

