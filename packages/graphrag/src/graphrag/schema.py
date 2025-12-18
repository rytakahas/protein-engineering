
from __future__ import annotations

from .db import Neo4jClient
from .utils import read_text


def apply_cql_file(db: Neo4jClient, cql_path: str) -> int:
    cql = read_text(cql_path)
    statements = [s.strip() for s in cql.split(";") if s.strip()]
    for st in statements:
        db.execute_cypher(st)
    return len(statements)
