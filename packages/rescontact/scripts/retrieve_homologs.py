#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieve homologs/MSAs (and optional PDB templates) from an MMseqs2 HTTP server.

Supports:
- FASTA input: one or more files via --fasta
- Dataset input: --source dataset with one or more --pdb-root directories
- Servers: https://a3m.mmseqs.com (recommended), https://api.colabfold.com (compatible)
- Output: JSON with alignment metadata (filled when --want-templates parses pdb70.m8)

Why fields were null before?
- a3m.mmseqs.com returns a tarball; alignment stats live in pdb70.m8.
  This script downloads and parses that tarball to fill pident, qstart, qend, evalue, bits.

Python 3.11 compatible. Standard library only (BioPython optional for robust PDB fallback, but not required).

Example:
  PYTHONPATH=src python scripts/retrieve_homologs.py \
    --fasta data/fasta/_subset.fa \
    --server-url https://a3m.mmseqs.com \
    --max-hits 16 \
    --want-templates \
    --qps 0.2 --inter-job-sleep 2 --max-retries 8 --timeout 1800 \
    --flush-every 1 \
    --out data/templates/mmseqs_hits.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import math
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from urllib import request, parse, error as urlerror
from pathlib import Path

# -----------------------
# Helpers: FASTA / PDB IO
# -----------------------

def read_fasta_many(files: List[str]) -> List[Tuple[str, str]]:
    seqs: List[Tuple[str, str]] = []
    for f in files:
        with open(f, "r") as fh:
            name = None
            buf = []
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith(">"):
                    if name is not None and buf:
                        seqs.append((name, "".join(buf)))
                    name = line[1:].strip() or f"seq_{len(seqs)+1}"
                    buf = []
                else:
                    buf.append(line.strip())
            if name is not None and buf:
                seqs.append((name, "".join(buf)))
    # de-dup by name
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for n, s in seqs:
        if n in seen:
            k = 2
            nn = f"{n}_{k}"
            while nn in seen:
                k += 1
                nn = f"{n}_{k}"
            n = nn
        seen.add(n)
        uniq.append((n, s))
    return uniq


# Minimal 3-letter to 1-letter map for SEQRES parsing (common residues)
AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","SEC":"U","PYL":"O"
}

def three_to_one_safe(tok: str) -> Optional[str]:
    tok = tok.upper()
    return AA3_TO_1.get(tok, None)

def collect_pdb_seqres(root: Path, limit_files: Optional[int], min_len: int) -> Iterator[Tuple[str, str]]:
    """
    Lightweight SEQRES reader that works even on symlinks. Avoids heavy Bio.PDB.
    Emits (query_id, sequence) for each chain meeting min_len.
    """
    count = 0
    for p in sorted(root.glob("*.pdb")):
        if limit_files is not None and count >= limit_files:
            break
        try:
            chains: Dict[str, List[str]] = {}
            with open(p, "r", errors="ignore") as fh:
                for line in fh:
                    if not line.startswith("SEQRES"):
                        continue
                    # columns per PDB spec; chain ID at col 12, residues start col 19
                    chain_id = line[11].strip() or "A"
                    toks = line[19:].split()
                    aa = []
                    for t in toks:
                        a1 = three_to_one_safe(t)
                        if a1:
                            aa.append(a1)
                        else:
                            # unknown code: skip (keeps alignment strict to known residues)
                            pass
                    chains.setdefault(chain_id, []).extend(aa)
            for cid, res in chains.items():
                seq = "".join(res)
                if len(seq) >= min_len:
                    qid = f"{p.stem}_{cid}"
                    yield (qid, seq)
                    count += 1
                    if limit_files is not None and count >= limit_files:
                        break
        except FileNotFoundError:
            # report and continue
            print(f"[WARN] missing file: {p}", file=sys.stderr)
        except Exception as e:
            print(f"[SEQRES] {p.name}: parse error: {e}", file=sys.stderr)

def collect_dataset_sequences(roots: List[Path], limit_files: Optional[int], min_len: int) -> List[Tuple[str, str]]:
    seqs: List[Tuple[str, str]] = []
    for root in roots:
        seqs.extend(list(collect_pdb_seqres(root, limit_files, min_len)))
    return seqs


# -----------------------
# MMseqs HTTP client
# -----------------------

@dataclass
class MMseqsResult:
    query_id: str
    length: int
    a3m_uniref: bool
    a3m_env: bool
    a3m_pdb: bool
    hits: List[Dict[str, Optional[float]]]

class MMseqsHTTPClient:
    def __init__(
        self,
        base_url: str,
        timeout: int = 1800,
        poll_interval: float = 2.0,
        qps: float = 0.0,
        inter_job_sleep: float = 0.0,
        max_retries: int = 6,
        verbose: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.qps = qps
        self.inter_job_sleep = inter_job_sleep
        self.max_retries = max_retries
        self.verbose = verbose
        self._last_request_ts = 0.0

    # --- low-level http with redirect + 429 backoff ----

    def _respect_qps(self):
        if self.qps and self.qps > 0:
            min_gap = 1.0 / self.qps
            now = time.time()
            wait = self._last_request_ts + min_gap - now
            if wait > 0:
                time.sleep(wait)

    def _open(self, req: request.Request) -> bytes:
        tries = 0
        backoff = 5.0
        while True:
            self._respect_qps()
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    self._last_request_ts = time.time()
                    return resp.read()
            except urlerror.HTTPError as e:
                code = getattr(e, "code", None)
                hdrs = getattr(e, "headers", {})
                loc = hdrs.get("Location")
                if code in (301, 302, 303, 307, 308) and loc:
                    # follow redirect manually to preserve POST for 307/308
                    if self.verbose:
                        print(f"[mmseqs] redirect {code} -> {loc}", file=sys.stderr)
                    req = request.Request(loc, data=req.data, headers=req.headers, method=req.get_method())
                    continue
                if code == 429:
                    retry_after = hdrs.get("Retry-After")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = backoff
                    else:
                        wait = backoff
                    wait = min(max(wait, 5.0), 120.0)
                    print(f"[mmseqs] 429 Too Many Requests. Sleeping {wait:.0f}s", file=sys.stderr)
                    time.sleep(wait)
                    backoff = min(backoff * 1.5, 300.0)
                    tries += 1
                    if tries > self.max_retries:
                        raise
                    continue
                raise
            except urlerror.URLError:
                tries += 1
                if tries > self.max_retries:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 300.0)

    def _post(self, path: str, form: Dict[str, str]) -> bytes:
        url = f"{self.base_url}{path}"
        data = parse.urlencode(form).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        return self._open(req)

    def _get(self, path: str) -> bytes:
        url = f"{self.base_url}{path}"
        req = request.Request(url, method="GET")
        return self._open(req)

    # --- ticket lifecycle ---

    def submit_msa(self, query_id: str, seq: str, use_env: bool, filtered: bool) -> str:
        """
        POST /ticket/msa
        """
        # a3m.mmseqs.com accepts: q (FASTA), mode (all/env), and filter (true/false)
        mode = "env" if use_env else "all"
        form = {
            "q": f">{query_id}\n{seq}\n",
            "mode": mode,
            "filter": "true" if filtered else "false",
        }
        raw = self._post("/ticket/msa", form)
        try:
            obj = json.loads(raw.decode("utf-8"))
            return obj["id"]
        except Exception:
            # If the server sent HTML (rate-limit page), raise a helpful error
            text = raw.decode("utf-8", errors="ignore")[:500]
            raise RuntimeError(f"Failed to get ticket id. Response head:\n{text}")

    def poll(self, ticket_id: str) -> Dict[str, str]:
        """
        GET /ticket/{id} until COMPLETE/ERROR.
        """
        t0 = time.time()
        while True:
            raw = self._get(f"/ticket/{ticket_id}")
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                text = raw.decode("utf-8", errors="ignore")[:300]
                obj = {"status": "", "raw": text}
            status = obj.get("status", "")
            if self.verbose:
                print(f"[mmseqs] status={status}", file=sys.stderr)
            if status == "COMPLETE":
                return obj
            if status == "ERROR":
                raise RuntimeError(f"Server returned ERROR for ticket {ticket_id}: {obj}")
            if time.time() - t0 > self.timeout:
                raise TimeoutError("Timed out waiting for A3M server result.")
            time.sleep(self.poll_interval)

    def download_result_tar(self, ticket_id: str) -> tarfile.TarFile:
        """
        GET /result/download/{id} -> in-memory tarfile
        """
        raw = self._get(f"/result/download/{ticket_id}")
        bio = io.BytesIO(raw)
        try:
            tf = tarfile.open(fileobj=bio, mode="r:gz")
        except tarfile.ReadError:
            # sometimes server gzips twice; try gunzip then untar
            bio.seek(0)
            ungz = gzip.GzipFile(fileobj=bio).read()
            tf = tarfile.open(fileobj=io.BytesIO(ungz), mode="r:*")
        return tf

    # --- parsing returned tarball ---

    @staticmethod
    def _read_member(tf: tarfile.TarFile, name: str) -> Optional[bytes]:
        try:
            m = tf.getmember(name)
        except KeyError:
            return None
        f = tf.extractfile(m)
        if not f:
            return None
        return f.read()

    @staticmethod
    def _parse_uniref_a3m(b: bytes, max_hits: int) -> List[Dict[str, Optional[float]]]:
        """Return subject-only list (no alignment stats for UniRef)."""
        if not b:
            return []
        hits = []
        for line in b.decode("utf-8", errors="ignore").splitlines():
            if line.startswith(">"):
                subj = line[1:].split()[0]
                if subj:  # de-dup consecutive duplicates
                    hits.append({
                        "subject": subj,
                        "pident": None,
                        "alnlen": None,
                        "qstart": None,
                        "qend": None,
                        "evalue": None,
                        "bits": None,
                    })
        # Drop the first header if it’s the query itself
        uniq = []
        seen = set()
        for h in hits:
            s = h["subject"]
            if s in seen:
                continue
            seen.add(s)
            uniq.append(h)
            if len(uniq) >= max_hits:
                break
        return uniq

    @staticmethod
    def _parse_pdb_m8(b: bytes, max_hits: int) -> List[Dict[str, Optional[float]]]:
        """
        Parse BLAST m8-like lines.
        Expected columns (standard): qid, sid, pident, length, mismatch, gapopen,
            qstart, qend, sstart, send, evalue, bitscore
        Some servers add extra columns; we take the first 12.
        """
        if not b:
            return []
        parsed: List[Dict[str, Optional[float]]] = []
        for line in b.decode("utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#"):
                continue
            cols = line.strip().split()
            if len(cols) < 12:
                # some variants start with an index, then sid, then qid... try to realign if possible
                # fallback: just record subject
                subj = cols[1] if len(cols) > 1 else cols[0]
                parsed.append({
                    "subject": subj,
                    "pident": None, "alnlen": None, "qstart": None, "qend": None,
                    "evalue": None, "bits": None,
                })
                continue
            qid, sid = cols[0], cols[1]
            try:
                pident = float(cols[2])
                alnlen = int(cols[3])
                qstart = int(cols[6])
                qend   = int(cols[7])
                evalue = float(cols[10])
                bits   = float(cols[11])
            except Exception:
                # defensive fallback
                pident = None; alnlen = None; qstart = None; qend = None; evalue = None; bits = None
            parsed.append({
                "subject": sid,
                "pident": pident,
                "alnlen": alnlen,
                "qstart": qstart,
                "qend": qend,
                "evalue": evalue,
                "bits": bits,
            })
        # keep best by evalue/bits; stable order
        parsed.sort(key=lambda h: (math.inf if h["evalue"] is None else h["evalue"],
                                   -1e9 if h["bits"] is None else -h["bits"]))
        uniq: List[Dict[str, Optional[float]]] = []
        seen = set()
        for h in parsed:
            s = h["subject"]
            if s in seen:
                continue
            seen.add(s)
            uniq.append(h)
            if len(uniq) >= max_hits:
                break
        return uniq

    def search_one(
        self,
        query_id: str,
        seq: str,
        max_hits: int,
        use_env: bool,
        filtered: bool,
        want_templates: bool,
    ) -> MMseqsResult:
        if self.verbose:
            print(f"[mmseqs] submit {query_id} (len={len(seq)})", file=sys.stderr)
        ticket = self.submit_msa(query_id, seq, use_env, filtered)
        self.poll(ticket)
        tf = self.download_result_tar(ticket)

        # files we may see
        a3m_bytes = self._read_member(tf, "uniref.a3m")
        # Template hits live here if server prepared them
        pdb_m8    = self._read_member(tf, "pdb70.m8") or self._read_member(tf, "pdb.m8")

        hits: List[Dict[str, Optional[float]]] = []
        a3m_uniref = bool(a3m_bytes)
        a3m_env = False  # could be flagged if you requested env; not strictly needed here
        a3m_pdb = bool(pdb_m8)

        if want_templates and pdb_m8:
            hits = self._parse_pdb_m8(pdb_m8, max_hits)
        else:
            hits = self._parse_uniref_a3m(a3m_bytes or b"", max_hits)

        # politeness between jobs
        if self.inter_job_sleep and self.inter_job_sleep > 0:
            time.sleep(self.inter_job_sleep)

        return MMseqsResult(
            query_id=query_id,
            length=len(seq),
            a3m_uniref=a3m_uniref,
            a3m_env=a3m_env,
            a3m_pdb=a3m_pdb,
            hits=hits,
        )


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", nargs="+", help="FASTA file(s)")
    src.add_argument("--source", choices=["dataset"], help="Use PDB dataset mode")

    ap.add_argument("--pdb-root", nargs="+", help="PDB root dir(s) (required if --source dataset)")
    ap.add_argument("--limit-files", type=int, default=None, help="Limit #PDB files to scan (dataset mode)")
    ap.add_argument("--min-len", type=int, default=30, help="Min sequence length to include (dataset mode)")

    ap.add_argument("--server-url", required=True, help="MMseqs server base URL (e.g. https://a3m.mmseqs.com)")
    ap.add_argument("--db", default="uniref", help="Ignored for a3m server, kept for compatibility")
    ap.add_argument("--max-hits", type=int, default=16)
    ap.add_argument("--use-env", action="store_true", help="Ask server for environmental sequences")
    ap.add_argument("--filtered", action="store_true", default=True, help="Request filtering on server")
    ap.add_argument("--want-templates", action="store_true", help="Parse pdb70.m8 to fill alignment stats")

    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--qps", type=float, default=0.0, help="Requests per second cap (client-side)")
    ap.add_argument("--inter-job-sleep", type=float, default=0.0, help="Sleep seconds between finished jobs")
    ap.add_argument("--max-retries", type=int, default=6)

    ap.add_argument("--flush-every", type=int, default=1, help="Write JSON after every N queries")
    ap.add_argument("--resume", action="store_true", help="Resume if OUT exists (skip seen query_ids)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    # Collect sequences
    if args.fasta:
        pairs = read_fasta_many(args.fasta)
        print(f"[retrieve_homologs] collected sequences: {len(pairs)} (FASTA)")
    else:
        if not args.pdb_root:
            ap.error("--pdb-root required when --source dataset")
        roots = [Path(p) for p in args.pdb_root]
        pairs = collect_dataset_sequences(roots, args.limit_files, args.min_len)
        if len(pairs) == 0:
            print("[retrieve_homologs] collected sequences: 0 (PDB)", file=sys.stderr)
        else:
            print(f"[retrieve_homologs] collected sequences: {len(pairs)} (PDB)")

    # Resume support
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: Dict[str, Dict] = {}
    if args.resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            for q in prev.get("queries", []):
                done[q["query_id"]] = q
            meta = prev.get("meta", {})
        except Exception:
            prev = None
        if args.verbose:
            print(f"[retrieve_homologs] resume: found {len(done)} existing queries", file=sys.stderr)

    client = MMseqsHTTPClient(
        base_url=args.server_url,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        qps=args.qps,
        inter_job_sleep=args.inter_job_sleep,
        max_retries=args.max_retries,
        verbose=args.verbose,
    )

    out_obj = {
        "meta": {
            "server_url": args.server_url,
            "db": args.db,
            "max_hits": args.max_hits,
            "min_ident": None,  # not enforced at server; can be post-filtered later
            "min_cov": None,    # same
            "use_env": bool(args.use_env),
            "filtered": bool(args.filtered),
            "want_templates": bool(args.want_templates),
        },
        "queries": [],
    }

    # Keep prior queries if resuming
    if args.resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            out_obj["queries"] = prev.get("queries", [])
        except Exception:
            pass

    flushed = 0
    for i, (qid, seq) in enumerate(pairs, 1):
        if qid in done:
            if args.verbose:
                print(f"[retrieve_homologs] skip existing {qid}", file=sys.stderr)
            continue
        try:
            res = client.search_one(
                qid, seq,
                max_hits=args.max_hits,
                use_env=args.use_env,
                filtered=args.filtered,
                want_templates=args.want_templates,
            )
            out_obj["queries"].append({
                "query_id": res.query_id,
                "length": res.length,
                "a3m_uniref": res.a3m_uniref,
                "a3m_env": res.a3m_env,
                "a3m_pdb": res.a3m_pdb,
                "hits": res.hits,
            })
        except Exception as e:
            print(f"[ERROR] {qid}: {e}", file=sys.stderr)
            out_obj["queries"].append({
                "query_id": qid,
                "length": len(seq),
                "a3m_uniref": False,
                "a3m_env": False,
                "a3m_pdb": False,
                "hits": [],
                "error": str(e),
            })

        if args.flush_every and (len(out_obj["queries"]) - flushed) >= args.flush_every:
            out_path.write_text(json.dumps(out_obj, ensure_ascii=False))
            flushed = len(out_obj["queries"])
            if args.verbose:
                print(f"[retrieve_homologs] flushed -> {out_path}", file=sys.stderr)

    # Final write
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False))
    print(f"[retrieve_homologs] wrote {args.out}  queries={len(out_obj['queries'])}")

if __name__ == "__main__":
    main()

