from __future__ import annotations

import csv
from typing import Dict, Any

from ..db import Neo4jClient


def ingest_ligands(db: Neo4jClient, csv_path: str) -> int:
    """
    Ingest Ligands from CSV.

    Expected columns (minimum):
      - ligand_id
    Optional:
      - name, smiles, inchi_key, scaffold_id, logp, tpsa, mw, source
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Light numeric casting
    norm: list[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        for k in ("logp", "tpsa", "mw"):
            if k in rr and rr[k] not in (None, "", "null"):
                try:
                    rr[k] = float(rr[k])
                except Exception:
                    pass
        norm.append(rr)

    cypher = """
    UNWIND $rows AS row
    MERGE (l:Ligand {ligand_id: row.ligand_id})
    SET l += row
    """
    if norm:
        db.run(cypher, params={"rows": norm})
    return len(norm)

