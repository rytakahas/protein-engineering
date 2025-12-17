from __future__ import annotations

import csv
from typing import Dict, Any

from ..db import Neo4jClient


def ingest_assays(db: Neo4jClient, csv_path: str) -> int:
    """
    Ingest AssayResult nodes + link to Ligand/Protein when IDs provided.

    Expected columns:
      - assay_id
    Optional:
      - ligand_id, uniprot_id, metric, value, units, source, doi, pmid
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    norm: list[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        if "value" in rr and rr["value"] not in (None, "", "null"):
            try:
                rr["value"] = float(rr["value"])
            except Exception:
                pass
        norm.append(rr)

    cypher = """
    UNWIND $rows AS row
    MERGE (a:AssayResult {assay_id: row.assay_id})
    SET a += row

    WITH row, a
    OPTIONAL MATCH (l:Ligand {ligand_id: row.ligand_id})
    FOREACH (_ IN CASE WHEN l IS NULL THEN [] ELSE [1] END |
      MERGE (l)-[r:HAS_ASSAY]->(a)
      SET r.metric = row.metric, r.value = row.value, r.units = row.units, r.source = row.source
    )

    WITH row, a
    OPTIONAL MATCH (p:Protein {uniprot_id: row.uniprot_id})
    FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
      MERGE (a)-[:MEASURES]->(p)
    )
    """
    if norm:
        db.run(cypher, params={"rows": norm})
    return len(norm)

