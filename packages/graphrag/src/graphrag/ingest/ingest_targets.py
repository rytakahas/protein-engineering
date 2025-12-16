from __future__ import annotations
import pandas as pd
from ..db import Neo4jClient


def ingest_targets(db: Neo4jClient, csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    required = {"uniprot_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"targets.csv missing columns: {sorted(missing)}")

    cypher = """
    MERGE (p:Protein {uniprot_id: $uniprot_id})
    SET p.name = coalesce($name, p.name),
        p.family = coalesce($family, p.family),
        p.organism = coalesce($organism, p.organism),
        p.gene = coalesce($gene, p.gene),
        p.sequence = coalesce($sequence, p.sequence),
        p.length = coalesce($length, p.length)
    """
    n = 0
    for _, row in df.iterrows():
        db.execute_cypher(cypher, row.to_dict())
        n += 1
    return {"ingested": n}

