from neo4j import GraphDatabase
from .config import Neo4jConfig

class Neo4jClient:
    def __init__(self, cfg: Neo4jConfig):
        self.cfg = cfg
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))

    def close(self):
        self.driver.close()

    def run(self, cypher: str, params: dict | None = None):
        params = params or {}
        with self.driver.session(database=self.cfg.database) as session:
            return list(session.run(cypher, params))

    def run_write(self, cypher: str, params: dict | None = None):
        params = params or {}
        with self.driver.session(database=self.cfg.database) as session:
            return session.execute_write(lambda tx: list(tx.run(cypher, params)))
