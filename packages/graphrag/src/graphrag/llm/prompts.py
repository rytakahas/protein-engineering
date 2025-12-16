from __future__ import annotations
from dataclasses import dataclass
from ..db import Neo4jClient


@dataclass
class SubgraphRetriever:
    db: Neo4jClient

    def retrieve(self, query_text: str, k: int = 50) -> dict:
        """
        Minimal retrieval:
        - Pull top-k nodes that match query text in name fields.
        - Then expand 1 hop to get edges around them.
        """
        # Match across a few common fields; adjust as your schema grows.
        cypher_seed = """
        MATCH (n)
        WHERE any(prop IN ['name','gene','family','uniprot_id','ligand_id','disease_id'] 
                  WHERE exists(n[prop]) AND toLower(toString(n[prop])) CONTAINS toLower($q))
        RETURN elementId(n) AS nid, labels(n) AS labels, properties(n) AS props
        LIMIT $k
        """
        seeds = self.db.query(cypher_seed, {"q": query_text, "k": k})

        seed_ids = [s["nid"] for s in seeds]
        if not seed_ids:
            return {"query": query_text, "seeds": [], "nodes": [], "edges": []}

        cypher_expand = """
        MATCH (a)-[r]->(b)
        WHERE elementId(a) IN $ids OR elementId(b) IN $ids
        RETURN elementId(a) AS src, labels(a) AS src_labels, properties(a) AS src_props,
               type(r) AS rel, properties(r) AS rel_props,
               elementId(b) AS dst, labels(b) AS dst_labels, properties(b) AS dst_props
        LIMIT $limit
        """
        rows = self.db.query(cypher_expand, {"ids": seed_ids, "limit": max(200, k * 10)})

        nodes = {}
        edges = []
        for r in rows:
            nodes[r["src"]] = {"id": r["src"], "labels": r["src_labels"], "props": r["src_props"]}
            nodes[r["dst"]] = {"id": r["dst"], "labels": r["dst_labels"], "props": r["dst_props"]}
            edges.append(
                {"src": r["src"], "dst": r["dst"], "type": r["rel"], "props": r["rel_props"]}
            )

        return {
            "query": query_text,
            "seeds": seeds,
            "nodes": list(nodes.values()),
            "edges": edges,
        }

