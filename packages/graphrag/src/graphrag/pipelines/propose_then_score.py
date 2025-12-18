
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import AppConfig
from ..db import Neo4jClient
from ..llm.loader import load_llm
from ..llm.prompts import render_prompt
from ..retrieval.subgraph_retriever import SubgraphRetriever
from ..utils import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = AppConfig.from_env()
    db = Neo4jClient(cfg.neo4j)
    retriever = SubgraphRetriever(db)
    llm = load_llm(cfg.llm)

    snapshot = retriever.retrieve(args.query, k=args.k, max_hops=1)

    prompt = render_prompt(
        "packages/graphrag/prompts/propose_candidates.md",
        variables={"question": args.query},
        snapshot=snapshot,
    )
    answer = llm.generate(prompt)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "retrieval_snapshot.json", snapshot)
    (out_dir / "llm_candidates.json").write_text(answer, encoding="utf-8")

    print(f"✅ wrote: {out_dir}/retrieval_snapshot.json")
    print(f"✅ wrote: {out_dir}/llm_candidates.json")


if __name__ == "__main__":
    main()
