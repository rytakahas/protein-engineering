from __future__ import annotations

import csv

from ..db import Neo4jClient


def ingest_contacts(db: Neo4jClient, csv_path: str) -> int:
    """
    Ingest residue-residue contacts as edges between Residue nodes for a given Protein.

    Expected columns:
      - uniprot_id, i, j
    Optional:
      - chain, aa_i, aa_j, w, dist
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # cast ints/floats
    for r in rows:
        for k in ("i", "j"):
            if k in r and r[k] not in (None, "", "null"):
                r[k] = int(float(r[k]))
        for k in ("w", "dist"):
            if k in r and r[k] not in (None, "", "null"):
                try:
                    r[k] = float(r[k])
                except Exception:
                    pass

    cypher = """
    UNWIND $rows AS row
    MERGE (p:Protein {uniprot_id: row.uniprot_id})

    MERGE (ri:Residue {residue_uid: row.uniprot_id + ':' + toString(row.i)})
    SET ri.uniprot_id = row.uniprot_id, ri.uniprot_pos = row.i, ri.aa = row.aa_i

    MERGE (rj:Residue {residue_uid: row.uniprot_id + ':' + toString(row.j)})
    SET rj.uniprot_id = row.uniprot_id, rj.uniprot_pos = row.j, rj.aa = row.aa_j

    MERGE (p)-[:HAS_RESIDUE]->(ri)
    MERGE (p)-[:HAS_RESIDUE]->(rj)

    MERGE (ri)-[c:CONTACTS]->(rj)
    SET c += row
    """
    if rows:
        db.run(cypher, params={"rows": rows})
    return len(rows)

