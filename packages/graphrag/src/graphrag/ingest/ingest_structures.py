from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

from ..db import Neo4jClient


def _derive_pdb_id(structure_id: str, path_str: str) -> str:
    # structure_id like "pdb_6AL4" -> "6AL4"
    if structure_id and structure_id.startswith("pdb_") and len(structure_id) > 4:
        return structure_id.split("_", 1)[1]

    # fallback: path like ".../6AL4.pdb" -> "6AL4"
    if path_str:
        stem = Path(path_str).name.split(".", 1)[0]
        if stem:
            return stem
    return ""


def ingest_structures(db: Neo4jClient, csv_path: str) -> int:
    """
    Accepts experimental or predicted structures as (:Structure).

    Your current structures.csv header example:
      model_id,uniprot_id,ligand_id,pdb_path,method,confidence,created_at

    We normalize to:
      structure_id (from structure_id or model_id or pdb_id)
      path (from path/pdb_path/mmcif_path)
      pdb_id (derived if possible)
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # IMPORTANT: Neo4j needs WITH between SET and (OPTIONAL) MATCH
    cypher = """
    UNWIND $rows AS row

    MERGE (s:Structure {structure_id: row.structure_id})
    SET s += row
    WITH row, s

    OPTIONAL MATCH (p:Protein  {uniprot_id: row.uniprot_id})
    OPTIONAL MATCH (l:Ligand   {ligand_id: row.ligand_id})
    OPTIONAL MATCH (ab:Antibody {antibody_id: row.antibody_id})
    OPTIONAL MATCH (pep:Peptide {peptide_id: row.peptide_id})

    FOREACH (_ IN CASE WHEN p   IS NULL THEN [] ELSE [1] END | MERGE (p)  -[:HAS_STRUCTURE]->(s))
    FOREACH (_ IN CASE WHEN l   IS NULL THEN [] ELSE [1] END | MERGE (l)  -[:HAS_STRUCTURE]->(s))
    FOREACH (_ IN CASE WHEN ab  IS NULL THEN [] ELSE [1] END | MERGE (ab) -[:HAS_STRUCTURE]->(s))
    FOREACH (_ IN CASE WHEN pep IS NULL THEN [] ELSE [1] END | MERGE (pep)-[:HAS_STRUCTURE]->(s))
    """

    norm_rows: list[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)

        sid = rr.get("structure_id") or rr.get("model_id") or rr.get("pdb_id")
        if not sid:
            sid = f"structure:{rr.get('uniprot_id','NA')}:{rr.get('ligand_id') or rr.get('antibody_id') or rr.get('peptide_id') or 'NA'}"
        rr["structure_id"] = sid

        # unify path columns
        rr["path"] = rr.get("path") or rr.get("pdb_path") or rr.get("mmcif_path") or ""

        # ensure pdb_id exists (prevents warnings and is useful later)
        rr["pdb_id"] = rr.get("pdb_id") or _derive_pdb_id(rr["structure_id"], rr["path"])

        # numeric cast
        if "confidence" in rr and rr["confidence"] not in (None, "", "null"):
            try:
                rr["confidence"] = float(rr["confidence"])
            except Exception:
                pass

        norm_rows.append(rr)

    if norm_rows:
        db.run(cypher, params={"rows": norm_rows})
    return len(norm_rows)

