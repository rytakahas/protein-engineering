
from __future__ import annotations

import csv
from typing import Dict, Any

from ..db import Neo4jClient


def ingest_toxicity(db: Neo4jClient, csv_path: str) -> int:
    """
    Your sample toxicity.csv columns:
      tox_id, ligand_id, flag, severity, source, notes
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cypher = """
    UNWIND $rows AS row
    MATCH (l:Ligand {ligand_id: row.ligand_id})
    MERGE (t:ToxicityEvent {tox_id: row.tox_id})
    SET t += row
    MERGE (l)-[:HAS_TOXICITY]->(t)
    """

    norm_rows: list[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        if not rr.get("ligand_id"):
            continue
        if not rr.get("tox_id"):
            rr["tox_id"] = f"tox:{rr['ligand_id']}:{rr.get('flag','unknown')}"
        norm_rows.append(rr)

    if norm_rows:
        db.run(cypher, params={"rows": norm_rows})
    return len(norm_rows)
