
from __future__ import annotations

from typing import Dict, Tuple, List


def _tokenize(q: str) -> List[str]:
    return [t.strip() for t in q.split() if t.strip()]


def build_generic_subgraph_query(query: str, k: int = 50, max_hops: int = 1) -> Tuple[str, Dict]:
    """
    Generic "search then expand 1-hop" Cypher query.
    Works without APOC.
    max_hops is accepted for compatibility; currently 1-hop expansion.
    """
    terms = _tokenize(query)

    props = [
        "name", "title", "description",
        "uniprot_id", "gene",
        "ligand_id", "antibody_id", "peptide_id",
        "structure_id", "pdb_id", "model_id",
        "disease_id",
        "doi", "pmid",
    ]

    cypher = """
    WITH $terms AS terms, $props AS props
    MATCH (n)
    WHERE any(term IN terms WHERE any(p IN props WHERE toLower(toString(n[p])) CONTAINS toLower(term)))
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

    params = {"terms": terms, "props": props, "k": int(k), "max_hops": int(max_hops)}
    return cypher, params
