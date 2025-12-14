#!/usr/bin/env bash
set -euo pipefail
docker compose -f packages/graphrag/docker-compose.neo4j.yml up -d
python -m graphrag init-db --constraints packages/graphrag/schema/constraints.cql
echo "Neo4j ready at http://localhost:7474 (neo4j/password)"
