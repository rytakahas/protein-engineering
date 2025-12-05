# scripts/mmseqs_msa_client.py
from __future__ import annotations
import json, time, math, sys
from typing import Dict, Iterable, Tuple, Optional, List
from urllib import request, parse, error

class A3MClient:
    def __init__(self, server_url: str, timeout: int = 600, qps: float = 0.2,
                 max_retries: int = 8, inter_job_sleep: float = 2.0,
                 user_agent: str = "rescontact/msa"):
        self.base = server_url.rstrip("/")
        self.timeout = timeout
        self.period = 1.0 / max(qps, 1e-6)
        self.max_retries = max_retries
        self.inter_job_sleep = inter_job_sleep
        self.hdrs = {"User-Agent": user_agent}

    def _post(self, path: str, data: Dict[str, str]) -> Dict:
        url = self.base + path
        payload = parse.urlencode(data).encode()
        req = request.Request(url, data=payload, headers=self.hdrs, method="POST")
        last = 0.0
        for attempt in range(self.max_retries + 1):
            # simple leaky-bucket rate limit
            now = time.time()
            wait = self.period - (now - last)
            if wait > 0: time.sleep(wait)
            last = time.time()
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status in (301, 302, 303, 307, 308):
                        url = resp.getheader("Location") or url
                        req = request.Request(url, data=payload, headers=self.hdrs, method="POST")
                        continue
                    return json.loads(resp.read().decode())
            except error.HTTPError as e:
                # Handle 307 (temp redirect) & 429 (rate limit)
                if e.code in (301,302,303,307,308):
                    redir = e.headers.get("Location")
                    if redir:
                        req = request.Request(redir, data=payload, headers=self.hdrs, method="POST")
                        continue
                if e.code == 429 and attempt < self.max_retries:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                raise
        raise RuntimeError("POST retries exceeded")

    def _get_json(self, path: str) -> Dict:
        url = self.base + path
        req = request.Request(url, headers=self.hdrs, method="GET")
        with request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _get_bytes(self, path: str) -> bytes:
        url = self.base + path
        req = request.Request(url, headers=self.hdrs, method="GET")
        with request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def submit_msa(self, query_id: str, seq: str, db: str = "uniref", mode: str = "all") -> str:
        # ColabFold-compatible MSA ticket
        data = {"q": f">{query_id}\n{seq}\n", "db": db, "mode": mode}
        r = self._post("/ticket/msa", data)
        return r["id"]

    def wait(self, ticket: str, poll: float = 2.0, max_wait: int = 36000) -> None:
        t0 = time.time()
        while True:
            r = self._get_json(f"/ticket/{ticket}")
            s = r.get("status", "")
            if s == "COMPLETE": return
            if s == "ERROR": raise RuntimeError(f"Server error for ticket {ticket}")
            if time.time() - t0 > max_wait: raise TimeoutError("A3M wait timed out")
            time.sleep(poll)

    def download_result_targz(self, ticket: str) -> bytes:
        return self._get_bytes(f"/result/download/{ticket}")

def run_batch(seqs: List[Tuple[str,str]], out_dir: str, server="https://a3m.mmseqs.com",
              db="uniref", qps=0.15) -> None:
    import os, io, tarfile, pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    cli = A3MClient(server, qps=qps)
    for qid, s in seqs:
        print(f"[msa] submit {qid} (L={len(s)})")
        ticket = cli.submit_msa(qid, s, db=db)
        time.sleep(cli.inter_job_sleep)
        cli.wait(ticket)
        blob = cli.download_result_targz(ticket)
        tgz_path = os.path.join(out_dir, f"{qid}.a3m.tgz")
        with open(tgz_path, "wb") as f: f.write(blob)
        # Optionally extract a3m directly:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            for m in tf.getmembers():
                if m.name.endswith(".a3m"):
                    with tf.extractfile(m) as f:
                        a3m = f.read().decode()
                    with open(os.path.join(out_dir, f"{qid}.a3m"), "w") as f:
                        f.write(a3m)
        print(f"[msa] saved {qid}.a3m (+tgz)")

