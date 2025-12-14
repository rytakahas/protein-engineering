import pandas as pd
from ..db import Neo4jClient

UPSERT_ASSAY = """
MERGE (a:AssayResult {assay_id:$assay_id})
SET a.uniprot_id=$uniprot_id, a.ligand_id=$ligand_id, a.peptide_id=$peptide_id,
    a.metric=$metric, a.value=$value, a.units=$units, a.system=$system,
    a.conditions=$conditions, a.source=$source
"""

BIND_EDGE = """
MATCH (l:Ligand {ligand_id:$ligand_id})
MATCH (p:Protein {uniprot_id:$uniprot_id})
MERGE (l)-[b:BINDS]->(p)
SET b.metric=$metric, b.value=$value, b.units=$units, b.source=$source
"""

def ingest_assays(db: Neo4jClient, path: str):
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        payload = {
            "assay_id": str(r["assay_id"]),
            "uniprot_id": str(r["uniprot_id"]),
            "ligand_id": str(r["ligand_id"]) if "ligand_id" in r and pd.notna(r["ligand_id"]) else None,
            "peptide_id": str(r["peptide_id"]) if "peptide_id" in r and pd.notna(r["peptide_id"]) else None,
            "metric": r.get("metric", ""),
            "value": float(r["value"]) if "value" in r and pd.notna(r["value"]) else None,
            "units": r.get("units", ""),
            "system": r.get("system", ""),
            "conditions": r.get("conditions", ""),
            "source": r.get("source", ""),
        }
        db.run_write(UPSERT_ASSAY, payload)
        if payload["ligand_id"]:
            db.run_write(BIND_EDGE, payload)
