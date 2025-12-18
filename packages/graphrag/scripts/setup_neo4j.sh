#!/usr/bin/env bash
set -euo pipefail

docker compose -f packages/graphrag/docker-compose.neo4j.yml up -d
echo "✅ Neo4j started"
echo "Browser: http://localhost:7474/browser"
echo "Auth: neo4j / neo4j"
