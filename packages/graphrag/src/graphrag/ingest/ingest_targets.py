import pandas as pd
from ..db import Neo4jClient

UPSERT_PROTEIN = """
MERGE (p:Protein {uniprot_id:$uniprot_id})
SET p.name=$name, p.family=$family, p.organism=$organism, p.gene=$gene,
    p.sequence=$sequence, p.length=$length
"""

def ingest_targets(db: Neo4jClient, path: str):
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        db.run_write(UPSERT_PROTEIN, {
            "uniprot_id": str(r["uniprot_id"]),
            "name": r.get("name", ""),
            "family": r.get("family", ""),
            "organism": r.get("organism", ""),
            "gene": r.get("gene", ""),
            "sequence": r.get("sequence", ""),
            "length": int(r["length"]) if "length" in r and pd.notna(r["length"]) else None,
        })
