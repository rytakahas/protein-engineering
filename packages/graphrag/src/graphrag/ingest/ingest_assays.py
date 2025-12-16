from __future__ import annotations
import pandas as pd
from ..db import Neo4jClient


def ingest_assays(db: Neo4jClient, csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    required = {"assay_id", "ligand_id", "uniprot_id", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"assays.csv missing columns: {sorted(missing)}")

    cypher = """
    MATCH (p:Protein {uniprot_id: $uniprot_id})
    MATCH (l:Ligand {ligand_id: $ligand_id})
    MERGE (a:AssayResult {assay_id: $assay_id})
    SET a.metric = $metric,
        a.value = $value,
        a.units = coalesce($units, a.units),
        a.source = coalesce($source, a.source),
        a.year = coalesce($year, a.year)
    MERGE (l)-[r:HAS_ASSAY]->(a)
    SET r.metric = $metric, r.value = $value, r.units = coalesce($units, r.units)
    MERGE (l)-[b:TARGETS]->(p)
    """
    n = 0
    for _, row in df.iterrows():
        db.execute_cypher(cypher, row.to_dict())
        n += 1
    return {"ingested": n}

