from __future__ import annotations
import pandas as pd
from ..db import Neo4jClient


def ingest_contacts(db: Neo4jClient, csv_path: str) -> dict:
    """
    Generic residue-residue contacts / interface contacts.

    Expected columns (minimum):
      uniprot_id, chain, i, aa_i, j, aa_j, w, dist
    Optional:
      model_id, contact_type
    """
    df = pd.read_csv(csv_path)
    required = {"uniprot_id", "i", "j"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"contacts.csv missing columns: {sorted(missing)}")

    cypher_res = """
    MERGE (p:Protein {uniprot_id: $uniprot_id})
    MERGE (ri:Residue {residue_uid: $uniprot_id + ':' + toString($i)})
    SET ri.uniprot_id = $uniprot_id, ri.uniprot_pos = $i, ri.aa = coalesce($aa_i, ri.aa), ri.chain = coalesce($chain, ri.chain)
    MERGE (rj:Residue {residue_uid: $uniprot_id + ':' + toString($j)})
    SET rj.uniprot_id = $uniprot_id, rj.uniprot_pos = $j, rj.aa = coalesce($aa_j, rj.aa), rj.chain = coalesce($chain, rj.chain)
    MERGE (p)-[:HAS_RESIDUE]->(ri)
    MERGE (p)-[:HAS_RESIDUE]->(rj)
    MERGE (ri)-[c:CONTACTS]->(rj)
    SET c.w = coalesce($w, 1.0),
        c.dist = coalesce($dist, null),
        c.model_id = coalesce($model_id, c.model_id),
        c.contact_type = coalesce($contact_type, c.contact_type)
    """
    n = 0
    for _, row in df.iterrows():
        db.execute_cypher(cypher_res, row.to_dict())
        n += 1
    return {"ingested": n}

