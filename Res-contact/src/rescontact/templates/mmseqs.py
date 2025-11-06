# src/rescontact/templates/mmseqs.py
from __future__ import annotations

import io
import json
import tarfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


@dataclass
class MMseqsHit:
    subject: str
    pident: Optional[float] = None
    alnlen: Optional[int] = None
    qstart: Optional[int] = None
    qend: Optional[int] = None
    qcov: Optional[float] = None
    evalue: Optional[float] = None
    bits: Optional[float] = None


class MMseqsA3MClient:
    """
    Minimal client for the a3m.mmseqs.com 'ticket' API used by ColabFold.

    Flow:
      POST   {server}/ticket/msa           data: q=">101\\nSEQ\\n>102\\nSEQ2...", mode={all|env|nofilter|...}
      GET    {server}/ticket/{id}          -> {"status": "PENDING|RUNNING|COMPLETE|ERROR|MAINTENANCE", ...}
      GET    {server}/result/download/{id} -> tar.gz with files like uniref.a3m, pdb70.m8, etc.
    """

    def __init__(self, server_url: str, timeout: int = 30):
        self.server_url = server_url.rstrip("/")
        self.sess = requests.Session()
        self.timeout = timeout

    # ---- low-level HTTP

    def _submit(self, seqs: List[str], mode: str) -> Dict:
        # Encode sequences as FASTA with numeric headers (101, 102, ...) to track them
        n0 = 101
        lines = []
        for i, s in enumerate(seqs):
            lines.append(f">{n0 + i}")
            lines.append(s)
        payload = {"q": "\n".join(lines) + "\n", "mode": mode}
        r = self.sess.post(f"{self.server_url}/ticket/msa", data=payload, timeout=self.timeout)
        try:
            return r.json()
        except Exception:
            return {"status": "UNKNOWN", "detail": r.text[:200]}

    def _status(self, ticket_id: str) -> Dict:
        r = self.sess.get(f"{self.server_url}/ticket/{ticket_id}", timeout=self.timeout)
        try:
            return r.json()
        except Exception:
            return {"status": "UNKNOWN", "detail": r.text[:200]}

    def _download(self, ticket_id: str) -> bytes:
        r = self.sess.get(f"{self.server_url}/result/download/{ticket_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.content

    # ---- public entrypoint

    def search_templates(
        self,
        sequences: List[str],
        use_env: bool = False,
        filtered: bool = True,
        poll_every: int = 5,
        max_wait_sec: int = 900,
    ) -> Dict[int, List[MMseqsHit]]:
        """
        Returns map: numeric_query_id -> list[MMseqsHit] (parsed from pdb70.m8).
        The numeric IDs are 101, 102, ... matching the order (with dedup preserved).
        """
        if not sequences:
            return {}

        # Deduplicate but keep original order mapping
        seqs_unique = []
        index_map = {}  # seq -> numeric id
        n0 = 101
        for s in sequences:
            if s not in seqs_unique:
                seqs_unique.append(s)
            # numeric ID for this sequence
        for i, s in enumerate(seqs_unique):
            index_map[s] = n0 + i
        Ms_for_original_order = [index_map[s] for s in sequences]

        # Mode selection (mirrors ColabFold’s colabfold.py)
        # filtered True -> {env|all}; filtered False -> {env-nofilter|nofilter}
        if filtered:
            mode = "env" if use_env else "all"
        else:
            mode = "env-nofilter" if use_env else "nofilter"

        out = self._submit(seqs_unique, mode)
        # Robust resubmission on soft statuses
        soft = {"UNKNOWN", "RATELIMIT"}
        while out.get("status") in soft:
            time.sleep(5)
            out = self._submit(seqs_unique, mode)

        if out.get("status") == "ERROR":
            raise RuntimeError("MMseqs2 API returned ERROR. Try later or check inputs.")
        if out.get("status") == "MAINTENANCE":
            raise RuntimeError("MMseqs2 API under maintenance. Try again soon.")

        ticket_id = out.get("id")
        if not ticket_id:
            raise RuntimeError(f"MMseqs2 API: missing ticket id in response: {json.dumps(out)[:200]}")

        # Poll
        waited = 0
        status = out.get("status", "PENDING")
        while status in {"PENDING", "RUNNING", "UNKNOWN"}:
            time.sleep(poll_every)
            waited += poll_every
            st = self._status(ticket_id)
            status = st.get("status", "UNKNOWN")
            if waited >= max_wait_sec and status != "COMPLETE":
                raise TimeoutError(f"MMseqs2 job did not complete within {max_wait_sec}s (status={status}).")

        if status != "COMPLETE":
            raise RuntimeError(f"MMseqs2 job finished in status={status} (expected COMPLETE).")

        # Download & parse tar.gz
        blob = self._download(ticket_id)
        hits_by_M: Dict[int, List[MMseqsHit]] = {}
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            # The PDB template alignments are in pdb70.m8 (BLAST m8-like)
            # Columns: qid subject %id alnlen mism gapopens qstart qend sstart send evalue bitscore
            m8 = None
            for member in tar.getmembers():
                if member.name.endswith("pdb70.m8"):
                    m8 = tar.extractfile(member)
                    break

            if m8 is not None:
                for raw in m8.read().decode("utf-8", errors="replace").splitlines():
                    if not raw.strip():
                        continue
                    parts = raw.split()
                    # Some servers prefix qid with the numeric marker (101, 102, ...)
                    try:
                        qid = int(parts[0])
                    except ValueError:
                        # If not numeric, skip (we rely on the numeric mapping)
                        continue
                    subj = parts[1]
                    try:
                        pident = float(parts[2])
                        alnlen = int(parts[3])
                        qstart = int(parts[6])
                        qend = int(parts[7])
                        evalue = float(parts[10])
                        bits = float(parts[11])
                    except Exception:
                        pident = alnlen = qstart = qend = None
                        evalue = bits = None

                    hit = MMseqsHit(
                        subject=subj,
                        pident=pident,
                        alnlen=alnlen,
                        qstart=qstart,
                        qend=qend,
                        qcov=None,  # can be computed later if needed
                        evalue=evalue,
                        bits=bits,
                    )
                    hits_by_M.setdefault(qid, []).append(hit)

        # Ensure we have keys for all original sequences, even if no hits
        for M in Ms_for_original_order:
            hits_by_M.setdefault(M, [])

        return hits_by_M


# Backward-compatible alias (your scripts import this name)
MMseqsRemoteClient = MMseqsA3MClient

