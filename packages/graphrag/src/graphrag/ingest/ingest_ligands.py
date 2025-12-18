
from __future__ import annotations

import csv
from typing import Dict, Any

from ..db import Neo4jClient


def ingest_ligands(db: Neo4jClient, csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cypher = """
    UNWIND $rows AS row
    MERGE (l:Ligand {ligand_id: row.ligand_id})
    SET l += row
    """

    norm_rows: list[Dict[str, Any]] = []
    for r in rows:
        if not r.get("ligand_id"):
            continue
        rr = dict(r)
        for k in ("logp", "tpsa", "mw"):
            if k in rr and rr[k] not in (None, "", "null"):
                try:
                    rr[k] = float(rr[k])
                except Exception:
                    pass
        norm_rows.append(rr)

    if norm_rows:
        db.run(cypher, params={"rows": norm_rows})
    return len(norm_rows)
