
from __future__ import annotations

from typing import Any, Dict

from ..db import Neo4jClient
from .cypher import build_generic_subgraph_query


class SubgraphRetriever:
    def __init__(self, db: Neo4jClient):
        self.db = db

    def retrieve(self, query: str, k: int = 50, max_hops: int = 1) -> Dict[str, Any]:
        cypher, params = build_generic_subgraph_query(query=query, k=int(k), max_hops=int(max_hops))
        rows = self.db.execute_cypher(cypher, params=params) or []
        nodes = rows[0].get("nodes", []) if rows else []
        edges = rows[0].get("edges", []) if rows else []

        return {
            "query": query,
            "k": int(k),
            "max_hops": int(max_hops),
            "nodes": nodes,
            "edges": edges,
        }
