import pandas as pd
from ..db import Neo4jClient

UPSERT_TOX = """
MERGE (t:ToxicityEvent {tox_id:$tox_id})
SET t.ligand_id=$ligand_id, t.tox_type=$tox_type, t.severity=$severity,
    t.evidence=$evidence, t.source=$source
"""

LINK_TOX = """
MATCH (l:Ligand {ligand_id:$ligand_id})
MATCH (t:ToxicityEvent {tox_id:$tox_id})
MERGE (l)-[:HAS_TOXICITY {source:$source}]->(t)
"""

def ingest_toxicity(db: Neo4jClient, path: str):
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        payload = {
            "tox_id": str(r["tox_id"]),
            "ligand_id": str(r["ligand_id"]),
            "tox_type": r.get("tox_type", ""),
            "severity": r.get("severity", ""),
            "evidence": r.get("evidence", ""),
            "source": r.get("source", ""),
        }
        db.run_write(UPSERT_TOX, payload)
        db.run_write(LINK_TOX, payload)
