
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
from pathlib import Path

from .config import AppConfig
from .db import Neo4jClient
from .schema import apply_cql_file
from .utils import write_json, read_json

from .llm.loader import load_llm
from .llm.prompts import render_prompt
from .retrieval.subgraph_retriever import SubgraphRetriever

from .ingest.ingest_targets import ingest_targets
from .ingest.ingest_ligands import ingest_ligands
from .ingest.ingest_assays import ingest_assays
from .ingest.ingest_toxicity import ingest_toxicity
from .ingest.ingest_structures import ingest_structures
from .ingest.ingest_contacts import ingest_contacts


def cmd_schema_apply(db: Neo4jClient, cql_path: str) -> None:
    n = apply_cql_file(db, cql_path)
    print(f"✅ Applied schema from: {cql_path} ({n} statements)")


def cmd_ingest(db: Neo4jClient, kind: str, input_path: str) -> None:
    if kind == "targets":
        n = ingest_targets(db, input_path)
    elif kind == "ligands":
        n = ingest_ligands(db, input_path)
    elif kind == "assays":
        n = ingest_assays(db, input_path)
    elif kind == "toxicity":
        n = ingest_toxicity(db, input_path)
    elif kind == "structures":
        n = ingest_structures(db, input_path)
    elif kind == "contacts":
        n = ingest_contacts(db, input_path)
    else:
        raise ValueError(f"Unknown ingest kind: {kind}")

    print(f"✅ Ingested {n} rows into kind='{kind}' from {input_path}")


def cmd_stats(db: Neo4jClient) -> None:
    q = """
    MATCH (n)
    UNWIND labels(n) AS lab
    RETURN lab AS label, count(*) AS n
    ORDER BY n DESC
    """
    rows = db.execute_cypher(q)
    print("=== Node counts by label ===")
    for r in rows:
        print(f"{r['label']}: {r['n']}")


def cmd_retrieve(db: Neo4jClient, query: str, k: int, out_path: str | None, max_hops: int = 1) -> None:
    retriever = SubgraphRetriever(db)
    snapshot = retriever.retrieve(query=query, k=int(k), max_hops=int(max_hops))

    if out_path:
        write_json(out_path, snapshot)
        print(f"✅ Wrote snapshot to: {out_path}")
    else:
        print(snapshot)


def cmd_answer(cfg: AppConfig, snapshot_path: str, prompt_path: str, out_path: str | None, question: str | None) -> None:
    snapshot = read_json(snapshot_path)
    llm = load_llm(cfg.llm)

    q = question or snapshot.get("query", "")
    prompt = render_prompt(prompt_path, variables={"question": q}, snapshot=snapshot)
    text = llm.generate(prompt)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text, encoding="utf-8")
        print(f"✅ Wrote answer to: {out_path}")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="graphrag.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # schema apply
    sp = sub.add_parser("schema", help="Schema tools")
    sp_sub = sp.add_subparsers(dest="schema_cmd", required=True)
    sp_apply = sp_sub.add_parser("apply", help="Apply constraints/indexes CQL")
    sp_apply.add_argument("--cql", required=True)

    # ingest
    ip = sub.add_parser("ingest", help="Ingest CSV data")
    ip.add_argument("kind", choices=["targets", "ligands", "assays", "toxicity", "structures", "contacts"])
    ip.add_argument("--input", required=True)

    # stats
    sub.add_parser("stats", help="Print node counts by label")

    # retrieve
    rp = sub.add_parser("retrieve", help="Retrieve a subgraph snapshot")
    rp.add_argument("--query", required=True)
    rp.add_argument("--k", type=int, default=50)
    rp.add_argument("--max-hops", type=int, default=1)
    rp.add_argument("--out", required=False)

    # answer
    ap2 = sub.add_parser("answer", help="Run LLM over a snapshot + prompt")
    ap2.add_argument("--snapshot", required=True)
    ap2.add_argument("--prompt", required=True)
    ap2.add_argument("--out", required=False)
    ap2.add_argument("--question", required=False)

    return ap


def main() -> None:
    cfg = AppConfig.from_env()
    db = Neo4jClient(cfg.neo4j)

    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "schema" and args.schema_cmd == "apply":
        cmd_schema_apply(db, args.cql)
        return

    if args.cmd == "ingest":
        cmd_ingest(db, args.kind, args.input)
        return

    if args.cmd == "stats":
        cmd_stats(db)
        return

    if args.cmd == "retrieve":
        cmd_retrieve(db, args.query, args.k, args.out, max_hops=args.max_hops)
        return

    if args.cmd == "answer":
        cmd_answer(cfg, args.snapshot, args.prompt, args.out, args.question)
        return


if __name__ == "__main__":
    main()
