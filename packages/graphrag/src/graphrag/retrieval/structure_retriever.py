# packages/graphrag/src/graphrag/retrieval/structure_retriever.py
from __future__ import annotations

from typing import Any, Dict

from ..db import Neo4jClient


def retrieve_structure_subgraph(
    db: Neo4jClient,
    *,
    uniprot_id: str,
    model_id: str,
    max_residues: int = 200,
    max_edges: int = 1500,
) -> Dict[str, Any]:
    """
    Protein + (optional) Structure + up to max_residues Residue nodes
    and up to max_edges CONTACTS edges for the given model_id.

    Trimming strategy:
      - hard caps on residues/edges
      - minimal property projection (much smaller prompts than properties(x))
    """

    cypher = """
    MATCH (p:Protein {uniprot_id: $uniprot_id})
    OPTIONAL MATCH (p)-[hs:HAS_STRUCTURE]->(s:Structure {structure_id: $model_id})

    // Collect residues participating in this model's contacts
    CALL (p) {
      OPTIONAL MATCH (p)-[:HAS_RESIDUE]->(r1:Residue)-[c:CONTACTS {model_id: $model_id}]->(r2:Residue)
      RETURN collect(DISTINCT r1) + collect(DISTINCT r2) AS rs
    }

    WITH p, s, hs, coalesce(rs, []) AS rs
    WITH p, s, hs, rs[0..$max_residues] AS rs

    // Collect up to max_edges CONTACTS among the selected residues (sorted by weight/dist)
    CALL (rs) {
      UNWIND rs AS a
      MATCH (a)-[c:CONTACTS {model_id: $model_id}]->(b:Residue)
      WHERE b IN rs
      WITH a, b, c
      ORDER BY coalesce(c.w, 1.0) DESC, coalesce(c.dist, 999999.0) ASC
      LIMIT $max_edges
      RETURN collect(DISTINCT {
        src: elementId(a),
        dst: elementId(b),
        type: type(c),
        props: {
          model_id: c.model_id,
          contact_type: c.contact_type,
          chain: c.chain,
          w: c.w,
          dist: c.dist
        }
      }) AS contact_edges
    }

    WITH p, s, hs, rs, coalesce(contact_edges, []) AS contact_edges

    // Keep query returning even if rs is empty
    UNWIND CASE WHEN size(rs)=0 THEN [NULL] ELSE rs END AS x
    WITH p, s, hs, contact_edges,
         collect(DISTINCT CASE
           WHEN x IS NULL THEN NULL
           ELSE {
             id: elementId(x),
             labels: labels(x),
             props: {
               residue_uid: x.residue_uid,
               uniprot_id: x.uniprot_id,
               pos: x.pos,
               aa: x.aa
             }
           }
         END) AS residue_nodes

    WITH p, s, hs, contact_edges,
         [n IN residue_nodes WHERE n IS NOT NULL] AS residue_nodes,
         CASE WHEN s IS NULL THEN [] ELSE [{
           id: elementId(s),
           labels: labels(s),
           props: {
             structure_id: s.structure_id,
             pdb_id: s.pdb_id,
             method: s.method,
             path: s.path,
             confidence: s.confidence,
             created_at: s.created_at
           }
         }] END AS s_nodes,
         CASE WHEN hs IS NULL OR s IS NULL THEN [] ELSE [{
           src: elementId(p),
           dst: elementId(s),
           type: type(hs),
           props: {}
         }] END AS s_edges

    RETURN
      [{
        id: elementId(p),
        labels: labels(p),
        props: {
          uniprot_id: p.uniprot_id,
          name: p.name,
          gene: p.gene,
          family: p.family,
          organism: p.organism,
          length: p.length
        }
      }] + residue_nodes + s_nodes AS nodes,
      contact_edges + s_edges AS edges
    """

    rows = db.execute_cypher(
        cypher,
        params={
            "uniprot_id": uniprot_id,
            "model_id": model_id,
            "max_residues": int(max_residues),
            "max_edges": int(max_edges),
        },
    ) or []

    nodes = rows[0].get("nodes", []) if rows else []
    edges = rows[0].get("edges", []) if rows else []

    return {
        "query": f"STRUCTURE uniprot={uniprot_id} model={model_id}",
        "uniprot_id": uniprot_id,
        "model_id": model_id,
        "max_residues": int(max_residues),
        "max_edges": int(max_edges),
        "nodes": nodes,
        "edges": edges,
    }

