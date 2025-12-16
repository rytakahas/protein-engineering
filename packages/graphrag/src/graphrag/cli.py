from __future__ import annotations
import argparse
from pathlib import Path

from .config import AppConfig
from .db import Neo4jClient
from .utils import read_text, write_json, read_json
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
    cql = read_text(cql_path)
    # naive split on ';' is risky for multiline; but OK for simple constraint files
    statements = [s.strip() for s in cql.split(";") if s.strip()]
    for st in statements:
        db.execute_cypher(st)
    print(f"✅ Applied schema from: {cql_path} ({len(statements)} statements)")


def cmd_ingest(db: Neo4jClient, kind: str, input_path: str) -> None:
    if kind == "targets":
        out = ingest_targets(db, input_path)
    elif kind == "ligands":
        out = ingest_ligands(db, input_path)
    elif kind == "assays":
        out = ingest_assays(db, input_path)
    elif kind == "toxicity":
        out = ingest_toxicity(db, input_path)
    elif kind == "structures":
        out = ingest_structures(db, input_path)
    elif kind == "contacts":
        out = ingest_contacts(db, input_path)
    else:
        raise SystemExit(f"Unknown ingest kind: {kind}")
    print(f"✅ Ingested {kind}: {out}")


def cmd_retrieve(db: Neo4jClient, query: str, k: int, out_path: str) -> None:
    retriever = SubgraphRetriever(db)
    snapshot = retriever.retrieve(query_text=query, k=k)
    write_json(out_path, snapshot)
    print(f"✅ Wrote retrieval snapshot: {out_path}")


def cmd_answer(cfg: AppConfig, snapshot_path: str, prompt_path: str, out_path: str) -> None:
    snapshot = read_json(snapshot_path)
    template = read_text(prompt_path)

    llm = load_llm(cfg.llm)

    context = {
        "question": snapshot.get("query", ""),
        "num_nodes": len(snapshot.get("nodes", [])),
        "num_edges": len(snapshot.get("edges", [])),
        "snapshot_json": snapshot,  # may be huge; prompts should select what they need
    }
    prompt = render_prompt(template, {
        "question": context["question"],
        "num_nodes": context["num_nodes"],
        "num_edges": context["num_edges"],
        "snapshot_json": str(context["snapshot_json"])[:200000],  # guardrail
    })

    text = llm.generate(prompt)
    write_json(out_path, {"question": context["question"], "response": text})
    print(f"✅ Wrote answer: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graphrag", description="GraphRAG CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # schema apply
    sp = sub.add_parser("schema", help="schema operations")
    sub2 = sp.add_subparsers(dest="schema_cmd", required=True)
    ap = sub2.add_parser("apply", help="apply schema CQL")
    ap.add_argument("--cql", required=True, help="Path to constraints/indexes .cql")

    # ingest
    ip = sub.add_parser("ingest", help="ingest CSVs into Neo4j")
    ip.add_argument("kind", choices=["targets", "ligands", "assays", "toxicity", "structures", "contacts"])
    ip.add_argument("--input", required=True)

    # retrieve
    rp = sub.add_parser("retrieve", help="retrieve a subgraph snapshot")
    rp.add_argument("--query", required=True)
    rp.add_argument("--k", type=int, default=50)
    rp.add_argument("--out", required=True)

    # answer
    an = sub.add_parser("answer", help="LLM summarize/rank using snapshot + prompt")
    an.add_argument("--snapshot", required=True)
    an.add_argument("--prompt", required=True)
    an.add_argument("--out", required=True)

    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = AppConfig.from_env()
    db = Neo4jClient.connect(cfg.neo4j)
    try:
        db.verify()

        if args.cmd == "schema" and args.schema_cmd == "apply":
            cmd_schema_apply(db, args.cql)

        elif args.cmd == "ingest":
            cmd_ingest(db, args.kind, args.input)

        elif args.cmd == "retrieve":
            cmd_retrieve(db, args.query, args.k, args.out)

        elif args.cmd == "answer":
            cmd_answer(cfg, args.snapshot, args.prompt, args.out)

        else:
            raise SystemExit("Unknown command")

    finally:
        db.close()


if __name__ == "__main__":
    main()

