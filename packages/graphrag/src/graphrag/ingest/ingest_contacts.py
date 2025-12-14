import pandas as pd
from ..db import Neo4jClient

UPSERT_RESIDUE = """
MERGE (r:Residue {residue_uid:$residue_uid})
SET r.uniprot_id=$uniprot_id, r.chain=$chain, r.index=$index, r.aa=$aa
"""

CONTACT_EDGE = """
MATCH (a:Residue {residue_uid:$a_uid})
MATCH (b:Residue {residue_uid:$b_uid})
MERGE (a)-[e:CONTACTS {source:$source}]->(b)
SET e.w=$w, e.dist=$dist
"""

def ingest_contacts(db: Neo4jClient, path: str, source: str = "rescontact"):
    """CSV expected columns:
    uniprot_id, chain, i, aa_i, j, aa_j, w, dist
    """
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        chain = r.get("chain", "A")
        uid_i = f"{r['uniprot_id']}:{chain}:{int(r['i'])}"
        uid_j = f"{r['uniprot_id']}:{chain}:{int(r['j'])}"

        db.run_write(UPSERT_RESIDUE, {
            "residue_uid": uid_i,
            "uniprot_id": str(r["uniprot_id"]),
            "chain": chain,
            "index": int(r["i"]),
            "aa": r.get("aa_i", "")
        })
        db.run_write(UPSERT_RESIDUE, {
            "residue_uid": uid_j,
            "uniprot_id": str(r["uniprot_id"]),
            "chain": chain,
            "index": int(r["j"]),
            "aa": r.get("aa_j", "")
        })

        db.run_write(CONTACT_EDGE, {
            "a_uid": uid_i,
            "b_uid": uid_j,
            "w": float(r["w"]) if "w" in r and pd.notna(r["w"]) else 1.0,
            "dist": float(r["dist"]) if "dist" in r and pd.notna(r["dist"]) else None,
            "source": source
        })
