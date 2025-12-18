# GraphRAG package (Neo4j)

GraphRAG = **Neo4j subgraph retrieval + LLM prompting** for ranking candidates (ligands / antibodies / peptides) and summarizing evidence.

This repo includes:
- **Ingestion** from CSVs into Neo4j (targets, ligands, assays, toxicity, structures, contacts)
- **Generic retrieval** (`retrieve`) + **structure-focused retrieval** (`retrieve-structure`)
- **Prompt rendering** with snapshot JSON injection
- **LLM backends**: HF Router (`hf_inference`), local HF (`hf_local`), or `noop` (no model call)

---

## 0) Start Neo4j

```bash
docker compose -f packages/graphrag/docker-compose.neo4j.yml up -d
```

Browser: http://localhost:7474/browser  
Auth: neo4j / neo4j

## 1) Environment variables

Create `.env` in repo root:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j
NEO4J_DATABASE=neo4j

# LLM options:
# 1) HF Router (remote)
LLM_PROVIDER=hf_inference
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
HF_TOKEN=YOUR_TOKEN

# 2) Local HF (optional)
# LLM_PROVIDER=hf_local
# LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
# LLM_DEVICE=auto

# 3) Noop (debug / offline)
# LLM_PROVIDER=noop
```

Load it in your shell before running commands:

```bash
set -a; source .env; set +a
```

## 2) Apply schema + ingest CSVs

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

## 3) Structure understanding layer (optional but recommended)

Your structures.csv can point to PDB/mmCIF paths. This layer can:

- extract sequences (sanity / mapping check)
- build residue–residue contacts from structures
- ingest those contacts into Neo4j for structure-based retrieval

### 3.1 Extract sequences from structure files

```bash
python -m graphrag.structure.extract_sequences \
  --structures data/graphrag/structures.csv \
  --out data/graphrag/structure_sequences.json
```

### 3.2 Build contacts from structures

```bash
python -m graphrag.structure.build_contacts \
  --structures data/graphrag/structures.csv \
  --targets data/graphrag/targets.csv \
  --out data/graphrag/contacts_from_structures.csv \
  --distance 8.0 \
  --contact-type intra_protein
```

### 3.3 Ingest structure-derived contacts

```bash
python -m graphrag.cli ingest contacts --input data/graphrag/contacts_from_structures.csv
```

## 4) Retrieve a generic subgraph snapshot

```bash
python -m graphrag.cli retrieve \
  --query "CD19" \
  --k 50 \
  --out data/graphrag/retrieval_snapshot.json

head -n 30 data/graphrag/retrieval_snapshot.json
```

## 5) Retrieve a structure-focused snapshot (Protein + Structure + Residues + CONTACTS)

This is useful when you want structure-aware summaries or constraints.

```bash
python -m graphrag.cli retrieve-structure \
  --uniprot P15391 \
  --model pdb_6AL4 \
  --max-residues 220 \
  --max-edges 700 \
  --out data/graphrag/snap_cd19_pdb6AL4.json

python -c "import json; s=json.load(open('data/graphrag/snap_cd19_pdb6AL4.json')); print(len(s['nodes']), len(s['edges']))"
```

## 6) Run an LLM prompt on a snapshot

### 6.1 Rank candidates (ligands / antibodies / peptides)

```bash
python -m graphrag.cli answer \
  --snapshot data/graphrag/retrieval_snapshot.json \
  --prompt packages/graphrag/prompts/propose_candidates.md \
  --out data/graphrag/candidates_ranked.json

head -n 60 data/graphrag/candidates_ranked.json
```

### 6.2 Summarize a structure subgraph (and avoid context overflow)

Structure snapshots can be large. Use `--max-nodes` / `--max-edges` to trim before sending to the LLM:

```bash
python -m graphrag.cli answer \
  --snapshot data/graphrag/snap_cd19_pdb6AL4.json \
  --prompt packages/graphrag/prompts/summarize_subgraph.md \
  --max-nodes 220 \
  --max-edges 700
```

## Common issues

### HF Router 403 (Cloudflare / provider block)

If you see HTTP 403 Forbidden and HTML from Cloudflare, your request is being blocked by the upstream provider behind the router.

Workarounds:
- switch to a different `HF_MODEL_ID`
- use `LLM_PROVIDER=hf_local`
- or temporarily use `LLM_PROVIDER=noop` to validate end-to-end prompts/snapshots

### HF Router 400 (context too long)

If you see an error like "maximum context length … reduce input tokens", trim:
- generate smaller snapshots (`--max-residues`, `--max-edges`)
- or use CLI trimming (`answer --max-nodes/--max-edges`)

### Sanity checks (Neo4j)

**Protein → Structure link:**

```bash
python -c "from graphrag.config import AppConfig; from graphrag.db import Neo4jClient; \
cfg=AppConfig.from_env(); db=Neo4jClient(cfg.neo4j); \
print(db.execute_cypher('MATCH (p:Protein)-[:HAS_STRUCTURE]->(s:Structure) RETURN p.uniprot_id, s.structure_id, s.pdb_id LIMIT 20'))"
```

**CONTACTS by model_id/contact_type:**

```bash
python -c "from graphrag.config import AppConfig; from graphrag.db import Neo4jClient; \
cfg=AppConfig.from_env(); db=Neo4jClient(cfg.neo4j); \
q='MATCH (:Residue)-[c:CONTACTS]->(:Residue) \
WITH coalesce(c.model_id,\"unknown\") AS model_id, coalesce(c.contact_type,\"unknown\") AS contact_type, count(*) AS n \
RETURN model_id, contact_type, n ORDER BY n DESC LIMIT 20'; \
print(db.execute_cypher(q))"
```

## Prompts

Prompts live in `packages/graphrag/prompts/` and use:
- `{{question}}`
- `{{snapshot_json}}`

Key prompts:
- `propose_candidates.md` — rank candidates
- `propose_constraints.md` — suggest residues/constraints (structure-aware)
- `summarize_subgraph.md` — summarize evidence
- `extract_entities.md` — entity extraction