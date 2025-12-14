import pandas as pd
from ..db import Neo4jClient

UPSERT_COMPLEX = """
MERGE (c:ComplexModel {model_id:$model_id})
SET c.uniprot_id=$uniprot_id, c.ligand_id=$ligand_id, c.peptide_id=$peptide_id,
    c.pdb_path=$pdb_path, c.method=$method, c.confidence=$confidence,
    c.created_at=$created_at
"""

def ingest_structures(db: Neo4jClient, path: str):
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        db.run_write(UPSERT_COMPLEX, {
            "model_id": str(r["model_id"]),
            "uniprot_id": str(r["uniprot_id"]),
            "ligand_id": str(r["ligand_id"]) if "ligand_id" in r and pd.notna(r["ligand_id"]) else None,
            "peptide_id": str(r["peptide_id"]) if "peptide_id" in r and pd.notna(r["peptide_id"]) else None,
            "pdb_path": r.get("pdb_path", ""),
            "method": r.get("method", ""),
            "confidence": float(r["confidence"]) if "confidence" in r and pd.notna(r["confidence"]) else None,
            "created_at": r.get("created_at", ""),
        })
