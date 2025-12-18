
from __future__ import annotations

import csv
from typing import Dict, Any

from ..db import Neo4jClient


def ingest_assays(db: Neo4jClient, csv_path: str) -> int:
    """
    Your sample assays.csv columns:
      assay_id, ligand_id, uniprot_id, metric, value, units, source, year

    This creates:
      (Ligand)-[:HAS_ASSAY]->(AssayResult)
      (Ligand)-[:TARGETS]->(Protein)
      (Ligand)-[:BINDS {metric, units, value}]->(Protein)
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cypher = """
    UNWIND $rows AS row

    MATCH (p:Protein {uniprot_id: row.uniprot_id})
    MATCH (l:Ligand {ligand_id: row.ligand_id})

    MERGE (a:AssayResult {assay_id: row.assay_id})
    SET a += row

    MERGE (l)-[:HAS_ASSAY]->(a)
    MERGE (l)-[:TARGETS]->(p)

    MERGE (l)-[b:BINDS]->(p)
    SET b.metric = row.metric,
        b.units = row.units,
        b.value = row.value,
        b.source = row.source,
        b.year = row.year
    """

    norm_rows: list[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        if not rr.get("assay_id"):
            rr["assay_id"] = f"assay:{rr.get('uniprot_id','NA')}:{rr.get('ligand_id','NA')}:{rr.get('metric','NA')}"
        if "value" in rr and rr["value"] not in (None, "", "null"):
            try:
                rr["value"] = float(rr["value"])
            except Exception:
                pass
        if "year" in rr and rr["year"] not in (None, "", "null"):
            try:
                rr["year"] = int(float(rr["year"]))
            except Exception:
                pass
        norm_rows.append(rr)

    if norm_rows:
        db.run(cypher, params={"rows": norm_rows})
    return len(norm_rows)
