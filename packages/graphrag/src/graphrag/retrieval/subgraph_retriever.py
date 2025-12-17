# packages/graphrag/src/graphrag/retrieval/subgraph_retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..db import Neo4jClient
from .cypher import build_generic_subgraph_query


@dataclass
class RetrievalResult:
    query: str
    k: int
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "k": self.k,
            "nodes": self.nodes,
            "edges": self.edges,
        }


class SubgraphRetriever:
    """
    Minimal subgraph retriever for GraphRAG.

    Strategy:
      1) Use a generic Cypher query that finds nodes whose text properties
         match query terms (name/id/description-like fields).
      2) Expand by 1 hop to include context edges.
      3) Return a compact snapshot (nodes + edges) for LLM prompting.

    This is intentionally simple and robust for demos.
    """

    def __init__(
        self,
        db: Neo4jClient,
        *,
        database: Optional[str] = None,
        max_hops: int = 1,
    ) -> None:
        self.db = db
        self.database = database
        self.max_hops = max_hops

    def retrieve(self, query: str, k: int = 50) -> Dict[str, Any]:
        cypher, params = build_generic_subgraph_query(query=query, k=k, max_hops=self.max_hops)
        rows = self.db.run(cypher, params=params, database=self.database)

        # Our cypher returns 1 row with nodes/edges arrays
        if not rows:
            result = RetrievalResult(query=query, k=k, nodes=[], edges=[])
            return result.to_snapshot()

        row0 = rows[0]
        nodes = row0.get("nodes", []) or []
        edges = row0.get("edges", []) or []

        result = RetrievalResult(query=query, k=k, nodes=nodes, edges=edges)
        return result.to_snapshot()

