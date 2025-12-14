import pandas as pd
from ..db import Neo4jClient

UPSERT_LIGAND = """
MERGE (l:Ligand {ligand_id:$ligand_id})
SET l.name=$name, l.smiles=$smiles, l.inchi_key=$inchi_key,
    l.scaffold_id=$scaffold_id, l.logp=$logp, l.tpsa=$tpsa, l.mw=$mw, l.source=$source
"""

def ingest_ligands(db: Neo4jClient, path: str):
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        db.run_write(UPSERT_LIGAND, {
            "ligand_id": str(r["ligand_id"]),
            "name": r.get("name", ""),
            "smiles": r.get("smiles", ""),
            "inchi_key": r.get("inchi_key", ""),
            "scaffold_id": r.get("scaffold_id", ""),
            "logp": float(r["logp"]) if "logp" in r and pd.notna(r["logp"]) else None,
            "tpsa": float(r["tpsa"]) if "tpsa" in r and pd.notna(r["tpsa"]) else None,
            "mw": float(r["mw"]) if "mw" in r and pd.notna(r["mw"]) else None,
            "source": r.get("source", ""),
        })
