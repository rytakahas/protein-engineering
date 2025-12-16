from __future__ import annotations
import pandas as pd
from ..db import Neo4jClient


def ingest_ligands(db: Neo4jClient, csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    required = {"ligand_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ligands.csv missing columns: {sorted(missing)}")

    cypher = """
    MERGE (l:Ligand {ligand_id: $ligand_id})
    SET l.name = coalesce($name, l.name),
        l.smiles = coalesce($smiles, l.smiles),
        l.inchi_key = coalesce($inchi_key, l.inchi_key),
        l.scaffold_id = coalesce($scaffold_id, l.scaffold_id),
        l.logp = coalesce($logp, l.logp),
        l.tpsa = coalesce($tpsa, l.tpsa),
        l.mw = coalesce($mw, l.mw),
        l.source = coalesce($source, l.source)
    """
    n = 0
    for _, row in df.iterrows():
        db.execute_cypher(cypher, row.to_dict())
        n += 1
    return {"ingested": n}

