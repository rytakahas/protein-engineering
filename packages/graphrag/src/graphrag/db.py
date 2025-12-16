from __future__ import annotations
from dataclasses import dataclass
from neo4j import GraphDatabase, Driver
from .config import Neo4jConfig


@dataclass
class Neo4jClient:
    cfg: Neo4jConfig
    driver: Driver

    @classmethod
    def connect(cls, cfg: Neo4jConfig) -> "Neo4jClient":
        driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        return cls(cfg=cfg, driver=driver)

    def close(self) -> None:
        self.driver.close()

    def verify(self) -> None:
        self.driver.verify_connectivity()

    def execute_cypher(self, cypher: str, params: dict | None = None) -> None:
        params = params or {}
        with self.driver.session(database=self.cfg.database) as s:
            s.run(cypher, params).consume()

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        params = params or {}
        with self.driver.session(database=self.cfg.database) as s:
            res = s.run(cypher, params)
            return [r.data() for r in res]

