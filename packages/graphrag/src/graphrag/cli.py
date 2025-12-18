from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import AppConfig
from .db import Neo4jClient
from .schema import apply_cql_file
from .utils import write_json, read_json

from .llm.loader import load_llm
from .llm.prompts import render_prompt
from .retrieval.subgraph_retriever import SubgraphRetriever

# structure-centric retriever (added by structure layer zip)
from .retrieval.structure_retriever import retrieve_structure_subgraph

from .ingest.ingest_targets import ingest_targets
from .ingest.ingest_ligands import ingest_ligands
from .ingest.ingest_assays import ingest_assays
from .ingest.ingest_toxicity import ingest_toxicity
from .ingest.ingest_structures import ingest_structures
from .ingest.ingest_contacts import ingest_contacts


def _compact_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def compact_snapshot_for_llm(
    snapshot: Dict[str, Any],
    *,
    max_nodes: int = 220,
    max_edges: int = 800,
) -> Dict[str, Any]:
    """
    Make snapshot LLM-friendly:
      - prune nodes/edges
      - shorten long Neo4j elementId() strings -> n0, n1, ...
      - drop huge properties (like Protein.sequence)
      - produce _snapshot_json_compact (string) used by prompts.py
    """
    nodes: List[Dict[str, Any]] = list(snapshot.get("nodes", []) or [])
    edges: List[Dict[str, Any]] = list(snapshot.get("edges", []) or [])

    def is_residue(n: Dict[str, Any]) -> bool:
        return "Residue" in (n.get("labels") or [])

    residue_nodes = [n for n in nodes if is_residue(n)]
    other_nodes = [n for n in nodes if not is_residue(n)]

    # Degree for residues (based on CONTACTS only)
    residue_ids = {n.get("id") for n in residue_nodes if n.get("id")}
    deg: Dict[str, int] = {}
    for e in edges:
        if e.get("type") != "CONTACTS":
            continue
        s, d = e.get("src"), e.get("dst")
        if s in residue_ids:
            deg[s] = deg.get(s, 0) + 1
        if d in residue_ids:
            deg[d] = deg.get(d, 0) + 1

    budget = max(0, int(max_nodes) - len(other_nodes))
    residue_nodes_sorted = sorted(residue_nodes, key=lambda n: deg.get(n.get("id", ""), 0), reverse=True)
    keep_residues = residue_nodes_sorted[:budget]
    keep_nodes = other_nodes + keep_residues

    keep_ids = {n.get("id") for n in keep_nodes if n.get("id")}
    kept_edges = [e for e in edges if e.get("src") in keep_ids and e.get("dst") in keep_ids]

    # Edge capping: keep all non-CONTACTS, cap CONTACTS by w desc, dist asc
    def edge_rank(e: Dict[str, Any]) -> Tuple[float, float]:
        props = e.get("props") or {}
        w = float(props.get("w", 1.0) or 1.0)
        dist = float(props.get("dist", 999999.0) or 999999.0)
        return (-w, dist)

    contact_edges = [e for e in kept_edges if e.get("type") == "CONTACTS"]
    other_edges = [e for e in kept_edges if e.get("type") != "CONTACTS"]
    contact_edges = sorted(contact_edges, key=edge_rank)[: int(max_edges)]
    kept_edges = other_edges + contact_edges

    # Remap node IDs to short IDs
    id_map: Dict[str, str] = {}
    for i, n in enumerate(keep_nodes):
        nid = n.get("id")
        if nid:
            id_map[nid] = f"n{i}"

    def slim_props(n: Dict[str, Any]) -> Dict[str, Any]:
        labels = n.get("labels") or []
        props = dict(n.get("props") or {})

        if "Protein" in labels:
            props.pop("sequence", None)
            keep = ["uniprot_id", "name", "gene", "family", "organism", "length"]
            return {k: props.get(k) for k in keep if k in props}

        if "Residue" in labels:
            keep = ["residue_uid", "uniprot_id", "pos", "aa"]
            return {k: props.get(k) for k in keep if k in props}

        if "Structure" in labels:
            keep = ["structure_id", "pdb_id", "path", "method", "confidence", "created_at"]
            return {k: props.get(k) for k in keep if k in props}

        # Default: cap keys
        out: Dict[str, Any] = {}
        for k in list(props.keys())[:25]:
            out[k] = props[k]
        return out

    out_nodes: List[Dict[str, Any]] = []
    for n in keep_nodes:
        nid = n.get("id", "")
        out_nodes.append(
            {
                "id": id_map.get(nid, nid),
                "labels": n.get("labels") or [],
                "props": slim_props(n),
            }
        )

    out_edges: List[Dict[str, Any]] = []
    for e in kept_edges:
        out_edges.append(
            {
                "src": id_map.get(e.get("src", ""), e.get("src")),
                "dst": id_map.get(e.get("dst", ""), e.get("dst")),
                "type": e.get("type"),
                "props": e.get("props") or {},
            }
        )

    compact = {
        # keep metadata keys if present
        "query": snapshot.get("query"),
        "k": snapshot.get("k"),
        "max_hops": snapshot.get("max_hops"),
        "uniprot_id": snapshot.get("uniprot_id"),
        "model_id": snapshot.get("model_id"),
        "max_residues": snapshot.get("max_residues"),
        "max_edges": snapshot.get("max_edges"),
        "nodes": out_nodes,
        "edges": out_edges,
    }
    compact["_snapshot_json_compact"] = _compact_dumps(compact)
    return compact


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


def cmd_retrieve_structure(
    db: Neo4jClient,
    *,
    uniprot_id: str,
    model_id: str,
    max_residues: int,
    max_edges: int,
    out_path: str | None,
) -> None:
    snapshot = retrieve_structure_subgraph(
        db,
        uniprot_id=uniprot_id,
        model_id=model_id,
        max_residues=int(max_residues),
        max_edges=int(max_edges),
    )

    if out_path:
        write_json(out_path, snapshot)
        print(f"✅ Wrote structure snapshot to: {out_path}")
    else:
        print(snapshot)


def cmd_answer(
    cfg: AppConfig,
    snapshot_path: str,
    prompt_path: str,
    out_path: str | None,
    question: str | None,
    *,
    compact: bool = True,
    max_nodes: int = 220,
    max_edges: int = 800,
) -> None:
    snapshot_raw = read_json(snapshot_path)

    snapshot = snapshot_raw
    if compact and isinstance(snapshot_raw, dict) and "nodes" in snapshot_raw and "edges" in snapshot_raw:
        snapshot = compact_snapshot_for_llm(snapshot_raw, max_nodes=max_nodes, max_edges=max_edges)

    llm = load_llm(cfg.llm)
    q = question or (snapshot.get("query", "") if isinstance(snapshot, dict) else "")
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

    # retrieve (generic)
    rp = sub.add_parser("retrieve", help="Retrieve a subgraph snapshot")
    rp.add_argument("--query", required=True)
    rp.add_argument("--k", type=int, default=50)
    rp.add_argument("--max-hops", type=int, default=1)
    rp.add_argument("--out", required=False)

    # retrieve-structure (structure-centric)
    rs = sub.add_parser("retrieve-structure", help="Retrieve a structure-centric snapshot (protein + residues + contacts)")
    rs.add_argument("--uniprot", required=True, help="Protein UniProt ID (e.g., P15391)")
    rs.add_argument("--model", required=True, help="Structure ID stored as Structure.structure_id (e.g., pdb_6AL4)")
    rs.add_argument("--max-residues", type=int, default=200)
    rs.add_argument("--max-edges", type=int, default=700)
    rs.add_argument("--out", required=False)

    # answer
    ap2 = sub.add_parser("answer", help="Run LLM over a snapshot + prompt")
    ap2.add_argument("--snapshot", required=True)
    ap2.add_argument("--prompt", required=True)
    ap2.add_argument("--out", required=False)
    ap2.add_argument("--question", required=False)

    # LLM safety knobs (avoid context overflow)
    ap2.add_argument("--no-compact", action="store_true", help="Disable snapshot compaction before LLM")
    ap2.add_argument("--max-nodes", type=int, default=220, help="Max nodes passed to LLM after compaction")
    ap2.add_argument("--max-edges", type=int, default=800, help="Max edges passed to LLM after compaction")

    return ap


def main() -> None:
    cfg = AppConfig.from_env()
    db = Neo4jClient(cfg.neo4j)

    try:
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

        if args.cmd == "retrieve-structure":
            cmd_retrieve_structure(
                db,
                uniprot_id=args.uniprot,
                model_id=args.model,
                max_residues=args.max_residues,
                max_edges=args.max_edges,
                out_path=args.out,
            )
            return

        if args.cmd == "answer":
            cmd_answer(
                cfg,
                args.snapshot,
                args.prompt,
                args.out,
                args.question,
                compact=(not args.no_compact),
                max_nodes=args.max_nodes,
                max_edges=args.max_edges,
            )
            return
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

