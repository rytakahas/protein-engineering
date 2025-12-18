from __future__ import annotations

import csv
from typing import Dict, Any

from ..db import Neo4jClient


def ingest_contacts(db: Neo4jClient, csv_path: str) -> int:
    """
    contacts.csv supported columns (superset):
      uniprot_id,chain,i,aa_i,j,aa_j,w,dist,model_id,contact_type

    Stores residues + CONTACTS relationships.
    Relationship properties:
      dist, w, chain, model_id, contact_type
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cypher = """
    UNWIND $rows AS row

    MERGE (p:Protein {uniprot_id: row.uniprot_id})

    MERGE (ri:Residue {residue_uid: row.uniprot_id + ':' + toString(row.i)})
    SET ri.uniprot_id = row.uniprot_id,
        ri.pos = row.i,
        ri.aa = row.aa_i

    MERGE (rj:Residue {residue_uid: row.uniprot_id + ':' + toString(row.j)})
    SET rj.uniprot_id = row.uniprot_id,
        rj.pos = row.j,
        rj.aa = row.aa_j

    MERGE (p)-[:HAS_RESIDUE]->(ri)
    MERGE (p)-[:HAS_RESIDUE]->(rj)

    MERGE (ri)-[c:CONTACTS]->(rj)
    SET c.dist = row.dist,
        c.w = row.w,
        c.chain = row.chain,
        c.model_id = coalesce(row.model_id, ''),
        c.contact_type = coalesce(row.contact_type, '')
    """

    norm_rows: list[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        if not rr.get("uniprot_id"):
            continue

        for k in ("i", "j"):
            if rr.get(k) not in (None, "", "null"):
                rr[k] = int(float(rr[k]))

        for k in ("dist", "w"):
            if rr.get(k) not in (None, "", "null"):
                rr[k] = float(rr[k])

        rr["model_id"] = rr.get("model_id") or ""
        rr["contact_type"] = rr.get("contact_type") or ""

        norm_rows.append(rr)

    if norm_rows:
        db.run(cypher, params={"rows": norm_rows})
    return len(norm_rows)
