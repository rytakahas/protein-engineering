from __future__ import annotations

import csv
from typing import Dict, Any

from ..db import Neo4jClient


def ingest_structures(db: Neo4jClient, csv_path: str) -> int:
    """
    Accepts experimental structures OR predicted structures (as Structure nodes).

    Your CSV columns:
      model_id, uniprot_id, ligand_id, pdb_path, method, confidence, created_at

    We map:
      structure_id := structure_id or model_id or pdb_id
      path := path or pdb_path or mmcif_path
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cypher = """
    UNWIND $rows AS row

    MERGE (s:Structure {structure_id: row.structure_id})
    SET s += row

    WITH row, s

    OPTIONAL MATCH (p:Protein {uniprot_id: row.uniprot_id})
    OPTIONAL MATCH (l:Ligand {ligand_id: row.ligand_id})
    OPTIONAL MATCH (ab:Antibody {antibody_id: row.antibody_id})
    OPTIONAL MATCH (pep:Peptide {peptide_id: row.peptide_id})

    FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
      MERGE (p)-[:HAS_STRUCTURE]->(s)
    )

    FOREACH (_ IN CASE WHEN l IS NULL THEN [] ELSE [1] END |
      MERGE (l)-[:HAS_STRUCTURE]->(s)
    )

    FOREACH (_ IN CASE WHEN ab IS NULL THEN [] ELSE [1] END |
      MERGE (ab)-[:HAS_STRUCTURE]->(s)
    )

    FOREACH (_ IN CASE WHEN pep IS NULL THEN [] ELSE [1] END |
      MERGE (pep)-[:HAS_STRUCTURE]->(s)
    )
    """

    norm_rows: list[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)

        sid = rr.get("structure_id") or rr.get("model_id") or rr.get("pdb_id")
        if not sid:
            sid = f"structure:{rr.get('uniprot_id','NA')}:{rr.get('ligand_id') or rr.get('antibody_id') or rr.get('peptide_id') or 'NA'}"
        rr["structure_id"] = sid

        if "path" not in rr:
            rr["path"] = rr.get("pdb_path") or rr.get("mmcif_path") or ""

        if "confidence" in rr and rr["confidence"] not in (None, "", "null"):
            try:
                rr["confidence"] = float(rr["confidence"])
            except Exception:
                pass

        norm_rows.append(rr)

    if norm_rows:
        db.run(cypher, params={"rows": norm_rows})
    return len(norm_rows)

