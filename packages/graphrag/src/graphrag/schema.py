from pathlib import Path
from .db import Neo4jClient

def apply_constraints(db: Neo4jClient, constraints_path: str):
    cql = Path(constraints_path).read_text(encoding="utf-8")
    for stmt in [s.strip() for s in cql.split(";") if s.strip()]:
        db.run_write(stmt)
