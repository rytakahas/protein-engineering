
from __future__ import annotations

from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver

from .config import Neo4jConfig


class Neo4jClient:
    """
    Small wrapper around neo4j.Driver.

    Construction options:
      - Neo4jClient(cfg) -> creates Driver internally
      - Neo4jClient(cfg, driver) -> uses provided Driver
    """

    def __init__(self, neo4j_cfg: Neo4jConfig, driver: Optional[Driver] = None):
        self.cfg = neo4j_cfg
        self._owns_driver = driver is None
        self.driver: Driver = driver or GraphDatabase.driver(
            neo4j_cfg.uri, auth=(neo4j_cfg.user, neo4j_cfg.password)
        )

    def close(self) -> None:
        if self._owns_driver and self.driver is not None:
            self.driver.close()

    def execute_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        with self.driver.session(database=self.cfg.database) as session:
            res = session.run(cypher, params)
            return [dict(r) for r in res]

    # Backward-compatible alias used by ingest scripts
    def run(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.execute_cypher(cypher, params=params)
