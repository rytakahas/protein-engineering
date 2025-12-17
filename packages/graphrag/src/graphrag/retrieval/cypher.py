# packages/graphrag/src/graphrag/retrieval/cypher.py
from __future__ import annotations

from typing import Dict, Tuple, List


def _tokenize(q: str) -> List[str]:
    return [t.strip() for t in q.split() if t.strip()]


def build_generic_subgraph_query(query: str, k: int = 50, max_hops: int = 1) -> Tuple[str, Dict]:
    """
    Build a generic "search then expand 1-hop" Cypher query.

    NOTE:
      - This version avoids APOC so it works on a default Neo4j docker image.
      - max_hops is kept for API compatibility (currently only 1 hop is implemented).
    """
    terms = _tokenize(query)

    # Properties we try to match (only if they exist on the node)
    props = [
        "name", "title", "description",
        "uniprot_id", "gene",
        "ligand_id", "antibody_id", "peptide_id",
        "pdb_id", "model_id",
        "doi", "pmid",
    ]

    cypher = """
    WITH $terms AS terms, $props AS props
    MATCH (n)
    WHERE any(term IN terms WHERE any(p IN props WHERE toString(n[p]) CONTAINS term))
    WITH DISTINCT n
    LIMIT $k
    OPTIONAL MATCH (n)-[r]-(m)
    WITH
      collect(DISTINCT {id: elementId(n), labels: labels(n), props: properties(n)}) +
      collect(DISTINCT {id: elementId(m), labels: labels(m), props: properties(m)}) AS nodes,
      collect(DISTINCT {
        src: elementId(startNode(r)),
        dst: elementId(endNode(r)),
        type: type(r),
        props: properties(r)
      }) AS edges
    RETURN nodes, edges
    """

    params = {"terms": terms, "props": props, "k": int(k)}
    return cypher, params

