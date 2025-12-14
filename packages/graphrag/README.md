# graphrag package

Neo4j-backed GraphRAG layer for the Protein Engineering Monorepo.

This package provides:
- Neo4j (Docker Compose) + schema constraints
- Ingestion scripts for targets/ligands/assays/tox/structures/contacts
- Subgraph retrieval (GraphRAG)
- LLM wrapper (Hugging Face local)
- Pipeline: propose candidates → (stub docking) → write back to Neo4j
- CLI entrypoint

## Quickstart

```bash
# from repo root (after installing -e packages/graphrag)
docker compose -f packages/graphrag/docker-compose.neo4j.yml up -d
graphrag init-db --constraints packages/graphrag/schema/constraints.cql

# then run proposal pipeline (requires Neo4j populated with at least Protein/Pocket/Ligand data)
graphrag propose-then-score --uniprot-id P12345 --pocket-id P12345:pk1
```

## Environment variables

Neo4j:
- NEO4J_URI (default bolt://localhost:7687)
- NEO4J_USER (default neo4j)
- NEO4J_PASSWORD (default password)
- NEO4J_DATABASE (default neo4j)

LLM:
- LLM_PROVIDER (default hf_local)
- LLM_MODEL (default Qwen/Qwen2.5-7B-Instruct)
- LLM_DEVICE (default auto)
- LLM_MAX_NEW_TOKENS (default 512)
- LLM_TEMPERATURE (default 0.2)
```
