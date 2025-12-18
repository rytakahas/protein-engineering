
from __future__ import annotations

import csv
from typing import Dict, Any

from ..db import Neo4jClient


def ingest_targets(db: Neo4jClient, csv_path: str) -> int:
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cypher = """
    UNWIND $rows AS row
    MERGE (p:Protein {uniprot_id: row.uniprot_id})
    SET p += row
    """

    norm_rows: list[Dict[str, Any]] = []
    for r in rows:
        if not r.get("uniprot_id"):
            continue
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
