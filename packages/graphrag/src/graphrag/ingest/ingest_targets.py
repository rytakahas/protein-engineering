from __future__ import annotations

import csv
from typing import Dict, Any

from ..db import Neo4jClient


def ingest_targets(db: Neo4jClient, csv_path: str) -> int:
    """
    Ingest Proteins (targets) from CSV.

    Expected columns (minimum):
      - uniprot_id
    Optional:
      - name, family, organism, gene, sequence, length
    """
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cypher = """
    UNWIND $rows AS row
    MERGE (p:Protein {uniprot_id: row.uniprot_id})
    SET p += row
    """

    # Normalize types lightly
    norm_rows: list[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        if "length" in rr and rr["length"] not in (None, "", "null"):
            try:
                rr["length"] = int(rr["length"])
            except Exception:
                pass
        norm_rows.append(rr)

    if norm_rows:
        db.run(cypher, params={"rows": norm_rows})
        count = len(norm_rows)
    return count

