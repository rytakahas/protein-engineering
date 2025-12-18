from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compact_dumps(obj: Any) -> str:
    # much smaller than indent=2
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def compact_snapshot_for_llm(
    snapshot: Dict[str, Any],
    *,
    max_nodes: int = 220,
    max_edges: int = 800,
) -> Dict[str, Any]:
    """
    Make snapshot JSON *LLM-friendly*:
      - keep Protein/Structure + top residues by degree
      - cap edges
      - drop heavy props (e.g., sequences)
      - replace long elementIds with short ids (n0, n1, ...)
    """
    nodes: List[Dict[str, Any]] = list(snapshot.get("nodes", []) or [])
    edges: List[Dict[str, Any]] = list(snapshot.get("edges", []) or [])

    # split residues vs others
    def is_residue(n: Dict[str, Any]) -> bool:
        return "Residue" in (n.get("labels") or [])

    residue_nodes = [n for n in nodes if is_residue(n)]
    other_nodes = [n for n in nodes if not is_residue(n)]

    # degree for residues based on CONTACTS edges
    deg: Dict[str, int] = {}
    residue_ids = {n["id"] for n in residue_nodes if "id" in n}
    for e in edges:
        if e.get("type") != "CONTACTS":
            continue
        s, d = e.get("src"), e.get("dst")
        if s in residue_ids:
            deg[s] = deg.get(s, 0) + 1
        if d in residue_ids:
            deg[d] = deg.get(d, 0) + 1

    # choose residues to keep
    budget = max(0, int(max_nodes) - len(other_nodes))
    residue_nodes_sorted = sorted(
        residue_nodes,
        key=lambda n: deg.get(n.get("id", ""), 0),
        reverse=True,
    )
    keep_residues = residue_nodes_sorted[:budget]
    keep_nodes = other_nodes + keep_residues

    keep_ids = {n["id"] for n in keep_nodes if "id" in n}

    # filter edges to kept nodes
    kept_edges = [e for e in edges if e.get("src") in keep_ids and e.get("dst") in keep_ids]

    # cap edges (prefer higher weight / shorter distance for CONTACTS)
    def edge_rank(e: Dict[str, Any]) -> Tuple[float, float]:
        props = e.get("props") or {}
        w = float(props.get("w", 1.0) or 1.0)
        dist = float(props.get("dist", 999999.0) or 999999.0)
        return (-w, dist)

    contact_edges = [e for e in kept_edges if e.get("type") == "CONTACTS"]
    non_contact_edges = [e for e in kept_edges if e.get("type") != "CONTACTS"]

    contact_edges = sorted(contact_edges, key=edge_rank)[: int(max_edges)]
    kept_edges = non_contact_edges + contact_edges

    # drop heavy props
    def slim_props(n: Dict[str, Any]) -> Dict[str, Any]:
        labels = n.get("labels") or []
        props = dict(n.get("props") or {})

        if "Protein" in labels:
            # sequences can be huge
            props.pop("sequence", None)
            keep = ["uniprot_id", "name", "gene", "family", "organism", "length"]
            return {k: props.get(k) for k in keep if k in props}

        if "Residue" in labels:
            keep = ["residue_uid", "uniprot_id", "pos", "aa"]
            return {k: props.get(k) for k in keep if k in props}

        if "Structure" in labels:
            keep = ["structure_id", "pdb_id", "path", "method", "confidence", "created_at"]
            return {k: props.get(k) for k in keep if k in props}

        # default: keep small props only
        # (cap number of keys to avoid surprises)
        out = {}
        for k in list(props.keys())[:25]:
            out[k] = props[k]
        return out

    # remap long ids -> short ids
    id_map: Dict[str, str] = {}
    for i, n in enumerate(keep_nodes):
        id_map[n["id"]] = f"n{i}"

    out_nodes: List[Dict[str, Any]] = []
    for n in keep_nodes:
        out_nodes.append(
            {
                "id": id_map.get(n["id"], n["id"]),
                "labels": n.get("labels") or [],
                "props": slim_props(n),
            }
        )

    out_edges: List[Dict[str, Any]] = []
    for e in kept_edges:
        out_edges.append(
            {
                "src": id_map.get(e["src"], e["src"]),
                "dst": id_map.get(e["dst"], e["dst"]),
                "type": e.get("type"),
                "props": e.get("props") or {},
            }
        )

    compact = {
        "query": snapshot.get("query"),
        "k": snapshot.get("k"),
        "max_hops": snapshot.get("max_hops"),
        "uniprot_id": snapshot.get("uniprot_id"),
        "model_id": snapshot.get("model_id"),
        "nodes": out_nodes,
        "edges": out_edges,
    }

    # also provide a pre-rendered compact JSON string if you want
    compact["_snapshot_json_compact"] = _compact_dumps(compact)
    return compact

