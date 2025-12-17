from __future__ import annotations

import csv

from ..db import Neo4jClient


def ingest_toxicity(db: Neo4jClient, csv_path: str) -> int:
    """
    Ingest ToxicityEvent nodes + link to Ligand when ligand_id provided.

    Expected columns:
      - tox_id
    Optional:
      - ligand_id, severity, description, source, doi, pmid
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cypher = """
    UNWIND $rows AS row
    MERGE (t:ToxicityEvent {tox_id: row.tox_id})
    SET t += row

    WITH row, t
    OPTIONAL MATCH (l:Ligand {ligand_id: row.ligand_id})
    FOREACH (_ IN CASE WHEN l IS NULL THEN [] ELSE [1] END |
      MERGE (l)-[r:HAS_TOXICITY]->(t)
      SET r.severity = row.severity, r.source = row.source
    )
    """
    if rows:
        db.run(cypher, params={"rows": rows})
    return len(rows)

