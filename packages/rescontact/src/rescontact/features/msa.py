
"""
MSA providers that do NOT require local FASTA databases.

- EbiJackhmmerProvider: uses the EMBL-EBI HMMER (jackhmmer) web service.
- NcbiBlastProvider: uses NCBI BLAST URLAPI (QBLAST).

Both are rate-limited and subject to upstream availability.
Provide contact info (email) and a descriptive user agent via kwargs.
"""

from __future__ import annotations
from typing import List, Optional
import time
import requests

class MSAProvider:
    def get_alignment(self, seq: str) -> List[str]:
        raise NotImplementedError

class EbiJackhmmerProvider(MSAProvider):
    def __init__(self, email: str, poll_interval: float = 4.0, session: Optional[requests.Session] = None, user_agent: Optional[str] = None):
        self.email = email
        self.poll = poll_interval
        self.sess = session or requests.Session()
        if user_agent:
            self.sess.headers.update({"User-Agent": user_agent})
        self.base = "https://www.ebi.ac.uk/Tools/hmmer/search/jackhmmer"

    def get_alignment(self, seq: str) -> List[str]:
        # Submit
        r = self.sess.post(self.base, data={"seq": seq, "email": self.email, "output": "json"})
        r.raise_for_status()
        job = r.json().get("uuid") or r.json().get("jobID") or r.json().get("id")
        if not job:
            raise RuntimeError("EBI jackhmmer: no job id returned")
        # Poll
        status_url = f"{self.base}/results/{job}"
        while True:
            s = self.sess.get(status_url, params={"format": "json"})
            if s.status_code == 404:
                time.sleep(self.poll); continue
            s.raise_for_status()
            js = s.json()
            if js.get("status") in ("RUNNING","PENDING"):
                time.sleep(self.poll); continue
            break
        # Try to fetch aligned sequences (FASTA or Stockholm)
        # Some deployments expose specific endpoints; here we try FASTA hits.
        fasta_url = f"{self.base}/results/{job}/aln-fasta"
        fr = self.sess.get(fasta_url)
        fr.raise_for_status()
        fasta = fr.text.strip().splitlines()
        seqs = []
        buf = []
        for line in fasta:
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf).replace("-", ""))
                    buf = []
            else:
                buf.append(line.strip())
        if buf:
            seqs.append("".join(buf).replace("-", ""))
        # fallback to at least include query itself
        if not seqs:
            seqs = [seq]
        return seqs

class NcbiBlastProvider(MSAProvider):
    def __init__(self, email: str, db: str = "swissprot", program: str = "blastp", poll_interval: float = 5.0, session: Optional[requests.Session] = None, user_agent: Optional[str] = None):
        self.email = email
        self.db = db
        self.program = program
        self.poll = poll_interval
        self.sess = session or requests.Session()
        if user_agent:
            self.sess.headers.update({"User-Agent": user_agent})
        self.base = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

    def get_alignment(self, seq: str) -> List[str]:
        # Submit
        put = self.sess.post(self.base, data={
            "CMD": "Put", "PROGRAM": self.program, "DATABASE": self.db,
            "QUERY": seq, "EMAIL": self.email
        })
        put.raise_for_status()
        rid = None
        for line in put.text.splitlines():
            if "RID =" in line:
                rid = line.split("=",1)[1].strip()
                break
        if not rid:
            raise RuntimeError("NCBI BLAST: RID not found")
        # Poll
        while True:
            get = self.sess.get(self.base, params={"CMD":"Get","RID":rid,"FORMAT_OBJECT":"Alignment"})
            if "Status=WAITING" in get.text:
                time.sleep(self.poll); continue
            if "Status=FAILED" in get.text:
                raise RuntimeError("NCBI BLAST job failed")
            if "Status=UNKNOWN" in get.text:
                time.sleep(self.poll); continue
            # Parse alignments (very rough extraction of subjects)
            lines = get.text.splitlines()
            seqs = []
            cur = []
            for ln in lines:
                if ln.startswith(">");
                    if cur:
                        seqs.append("".join(cur).replace("-", ""))
                        cur = []
                if ln.startswith("Sbjct"):
                    parts = ln.split()
                    if len(parts)>=3:
                        cur.append(parts[2])
            if cur:
                seqs.append("".join(cur).replace("-", ""))
            if not seqs:
                seqs = [seq]
            return seqs

def msa_to_pssm(msa: List[str], alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> "np.ndarray":
    """Convert list of aligned sequences into frequency matrix [L, 20]."""
    import numpy as np
    if not msa:
        raise ValueError("Empty MSA")
    L = len(msa[0])
    A = {aa:i for i,aa in enumerate(alphabet)}
    mat = np.zeros((L, len(alphabet)), dtype=np.float32)
    for s in msa:
        for i, ch in enumerate(s[:L]):  # truncate/assume equal length
            if ch in A:
                mat[i, A[ch]] += 1.0
    mat /= max(1.0, len(msa))
    return mat
