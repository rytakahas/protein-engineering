
# GraphRAG package (Neo4j)

GraphRAG = Neo4j retrieval + LLM ranking (HF Router or local).

## Quick start

```bash
docker compose -f packages/graphrag/docker-compose.neo4j.yml up -d

# Browser: http://localhost:7474/browser
# Auth: neo4j / neo4j
```

Create `.env` in repo root:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j
NEO4J_DATABASE=neo4j

LLM_PROVIDER=hf_inference
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=YOUR_TOKEN
```

Apply schema + ingest demo CSVs:

```bash
python -m graphrag.cli schema apply --cql packages/graphrag/schema/constraints.cql

python -m graphrag.cli ingest targets    --input data/graphrag/targets.csv
python -m graphrag.cli ingest ligands    --input data/graphrag/ligands.csv
python -m graphrag.cli ingest assays     --input data/graphrag/assays.csv
python -m graphrag.cli ingest toxicity   --input data/graphrag/toxicity.csv
python -m graphrag.cli ingest structures --input data/graphrag/structures.csv
python -m graphrag.cli ingest contacts   --input data/graphrag/contacts.csv

python -m graphrag.cli stats
```

Retrieve:

```bash
python -m graphrag.cli retrieve --query "CD19" --k 50 --out data/graphrag/retrieval_snapshot.json
```

Answer:

```bash
python -m graphrag.cli answer \
  --snapshot data/graphrag/retrieval_snapshot.json \
  --prompt packages/graphrag/prompts/propose_candidates.md \
  --out data/graphrag/candidates_ranked.json
```
