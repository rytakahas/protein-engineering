from __future__ import annotations

import csv

from ..db import Neo4jClient


def ingest_structures(db: Neo4jClient, csv_path: str) -> int:
    """
    Ingest Structure (experimental) or ComplexModel (predicted) records.

    Expected columns:
      - model_id
    Optional:
      - pdb_id, uniprot_id, ligand_id, antibody_id, peptide_id
      - pdb_path, mmcif_path, method, confidence, created_at
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cypher = """
    UNWIND $rows AS row
    // We treat everything as Structure unless method hints predicted;
    // you can split later if you want.
    MERGE (s:Structure {model_id: row.model_id})
    SET s += row

    WITH row, s
    OPTIONAL MATCH (p:Protein {uniprot_id: row.uniprot_id})
    FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
      MERGE (p)-[:HAS_STRUCTURE]->(s)
    )

    WITH row, s
    OPTIONAL MATCH (l:Ligand {ligand_id: row.ligand_id})
    FOREACH (_ IN CASE WHEN l IS NULL THEN [] ELSE [1] END |
      MERGE (l)-[:HAS_STRUCTURE]->(s)
    )
    """
    if rows:
        db.run(cypher, params={"rows": rows})
    return len(rows)

