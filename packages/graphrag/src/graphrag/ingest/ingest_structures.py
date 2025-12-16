from __future__ import annotations
import pandas as pd
from ..db import Neo4jClient


def ingest_structures(db: Neo4jClient, csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    required = {"model_id", "uniprot_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"structures.csv missing columns: {sorted(missing)}")

    cypher = """
    MATCH (p:Protein {uniprot_id: $uniprot_id})
    MERGE (s:Structure {structure_id: $model_id})
    SET s.path = coalesce($pdb_path, s.path),
        s.method = coalesce($method, s.method),
        s.confidence = coalesce($confidence, s.confidence),
        s.created_at = coalesce($created_at, s.created_at)
    MERGE (p)-[:HAS_STRUCTURE]->(s)
    WITH s
    OPTIONAL MATCH (l:Ligand {ligand_id: $ligand_id})
    FOREACH (_ IN CASE WHEN l IS NULL THEN [] ELSE [1] END |
      MERGE (l)-[:HAS_STRUCTURE]->(s)
    )
    """
    n = 0
    for _, row in df.iterrows():
        db.execute_cypher(cypher, row.to_dict())
        n += 1
    return {"ingested": n}

