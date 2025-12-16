# GraphRAG Package (Neo4j)

**Target → Candidate → Evidence Retrieval**  
*WT complexes from AlphaFold/Diffusion supported*

---

## Overview

This package provides the **knowledge + decision layer** for the `protein-engineering` monorepo.

**GraphRAG = Knowledge Graph retrieval (Neo4j) + LLM summarization / ranking.**

### Purpose

Connect and query relationships between:
- **Targets** ↔ **Diseases** ↔ **Drugs/Ligands** ↔ **Antibodies/Peptides**
- **Structures** (experimental or predicted) ↔ **Assays** ↔ **Toxicity** ↔ **Literature**

Retrieve **small, relevant subgraphs** and use LLM/ranking to produce:
- **Ranked candidates** (ligands / antibodies / peptides)
- **Design constraints** (pocket/epitope residues, must-contact lists, liability avoid-lists)

---

## ⚠️ Important Clarification

**GraphRAG does NOT compute binding geometry** (no docking poses, no ΔG calculations).

### Geometry Sources

Binding geometry can come from:
- **WT complex predicted by AlphaFold / diffusion** (PDB/mmCIF)
- Existing experimental structures (PDB / SAbDab / PDBbind)
- External docking / complex predictors (optional integration)

**GraphRAG evaluates** in terms of:
- Evidence strength
- Similarity metrics
- Constraint satisfaction

**NOT** physics-based energy calculations.

---

## What You Can Do

### Example Questions

1. "For target X in leukemia, what are the best known ligands/antibodies and what evidence supports them?"
2. "Which candidates avoid known toxicity liabilities but still bind related targets?"
3. "Given a **WT complex predicted by AF/diffusion**, what residues define the pocket/epitope and what constraints should we enforce for the next design round?"
4. "Which known complexes are **most similar** to my WT predicted interface (pocket/epitope fingerprint)?"

### Outputs

- **Ranked candidate list** (ligands / antibodies / peptides)
- **Structured constraints**:
  - Pocket/epitope residues
  - Must-contact residues
  - Avoid-list liabilities
- **Retrieval snapshot** (subgraph JSON for reproducibility)
- **Similar complex list** (by interface similarity to your WT predicted complex)

---

## Data Sources (Recommended)

Build a **project-scoped knowledge graph** by combining:

| Source | Content |
|--------|---------|
| **Open Targets Platform API** | Target–disease evidence, drug/target context |
| **SAbDab** | Antibody–antigen complexes (benchmark/reference) |
| **PDBbind** | Protein–ligand complexes + affinity data |
| **ChEMBL** | Ligand bioactivity data |
| **PubMed/Papers** | Literature evidence |

> 💡 **Tip**: You don't need a full global KG. Build a **target-scoped subgraph** and enrich it iteratively.

---

## WT Complexes from AlphaFold / Diffusion

### Use Case

If your WT complex comes from **AlphaFold / diffusion** and is saved as **PDB or mmCIF**, GraphRAG can work **without docking** by storing:

### Data Model

**ComplexModel** node:
- `model_id`: Unique identifier
- `method`: `alphafold`, `diffusion`, `rosettafold`, etc.
- `confidence`: Model confidence score (if available)
- `pdb_path` / `mmcif_path`: File location

**Interface annotations**:
- `Pocket` residues (protein residues contacting ligand)
- `Epitope` residues (protein residues contacting antibody/peptide)
- `InterfaceResidueSet` fingerprint (residue list + optional hashed signature)

### Residue Numbering

Since your numbering is **consistent with UniProt**, store residues as:

```cypher
(Protein {uniprot_id})-[:HAS_RESIDUE]->(Residue {uniprot_pos})
```

### Similarity Retrieval

Enables retrieval based on:
- ✅ Pocket/epitope residue overlap
- ✅ Target family similarity
- ✅ Ligand scaffold similarity
- ✅ Antibody class / CDR motif similarity (if sequences stored)

---

## Directory Layout

```
packages/graphrag/
├── README.md
├── pyproject.toml
├── docker-compose.neo4j.yml
├── schema/
│   ├── nodes.yaml
│   ├── edges.yaml
│   └── constraints.cql
├── prompts/
│   ├── extract_entities.md
│   ├── summarize_subgraph.md
│   ├── propose_candidates.md
│   └── propose_constraints.md
├── scripts/
│   ├── setup_neo4j.sh
│   └── load_example.sh
└── src/graphrag/
    ├── cli.py
    ├── config.py
    ├── db.py
    ├── schema.py
    ├── utils.py
    ├── llm/
    │   ├── base.py
    │   ├── loader.py
    │   ├── prompts.py
    │   └── hf_local.py
    ├── ingest/
    │   ├── ingest_targets.py
    │   ├── ingest_ligands.py
    │   ├── ingest_assays.py
    │   ├── ingest_toxicity.py
    │   ├── ingest_structures.py    # Experimental structures (PDB/SAbDab/PDBbind)
    │   └── ingest_contacts.py      # Interface contacts / residue sets
    ├── retrieval/
    │   ├── cypher.py
    │   └── subgraph_retriever.py
    └── pipelines/
        └── propose_then_score.py
```

---

## Quick Start

### 1. Start Neo4j (Local)

From repository root:

```bash
docker compose -f packages/graphrag/docker-compose.neo4j.yml up -d
```

**Access points**:
- Neo4j Browser: `http://localhost:7474`
- Bolt connection: `bolt://localhost:7687`
- Default credentials: `neo4j` / `password`

### 2. Install GraphRAG Package

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e packages/graphrag
```

### 3. Configure Environment

```bash
# Neo4j connection
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export NEO4J_DATABASE="neo4j"

# LLM configuration (Hugging Face)
export LLM_PROVIDER="hf_local"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LLM_DEVICE="auto"
export LLM_MAX_NEW_TOKENS="512"
export LLM_TEMPERATURE="0.2"
```

---

## Building the Knowledge Graph

### Minimal Node Types

| Node Type | Key Property | Description |
|-----------|--------------|-------------|
| `Protein` | `uniprot_id` | UniProt protein entry |
| `Disease` | `efo_id` or `doid` | Disease ontology ID |
| `Ligand` | `ligand_id` | Small molecule (ChEMBL/PubChem/SMILES) |
| `Antibody` | `antibody_id` | Antibody + optional VH/VL/CDR sequences |
| `Peptide` | `peptide_id` | Peptide binder + sequence |
| `Structure` | `pdb_id` | Experimental structure (PDB/SAbDab/PDBbind) |
| `ComplexModel` | `model_id` | Predicted complex (AF/diffusion output) |
| `AssayResult` | `assay_id` | Binding affinity (IC50/Kd/Ki) |
| `ToxicityEvent` | `tox_id` | Toxicity flag + evidence |
| `Paper` | `doi` or `pmid` | Literature reference |
| `Residue` | `residue_uid` | Protein residue (UniProt position) |
| `Pocket` | `pocket_id` | Binding pocket (residue set) |
| `Epitope` | `epitope_id` | Antibody epitope (residue set) |

### Minimal Edge Types

| Edge | From | To | Properties |
|------|------|-----|-----------|
| `TARGETS` | Drug/Ligand/Antibody | Protein | `source` |
| `ASSOCIATED_WITH` | Protein | Disease | `evidence_score` |
| `HAS_STRUCTURE` | Protein/Ligand/Antibody | Structure | `method` |
| `HAS_MODEL` | Protein/Ligand/Antibody | ComplexModel | `confidence` |
| `HAS_ASSAY` | Ligand/Antibody | AssayResult | `metric`, `value` |
| `HAS_TOXICITY` | Ligand | ToxicityEvent | `severity` |
| `MENTIONED_IN` | Any | Paper | `snippet` |
| `HAS_RESIDUE` | Protein | Residue | - |
| `HAS_POCKET` | ComplexModel/Structure | Pocket | `method` |
| `HAS_EPITOPE` | ComplexModel/Structure | Epitope | `method` |
| `COMPOSED_OF` | Pocket/Epitope | Residue | `role` |
| `BINDS` | Ligand/Peptide | Protein/Pocket | `metric`, `value` |

---

## Data Ingestion

### 1. Create Constraints / Indexes

```bash
graphrag init-db --constraints packages/graphrag/schema/constraints.cql
```

### 2. Ingest Target/Disease Evidence

Prepare `data/graphrag/targets.csv`:

```csv
uniprot_id,name,family,organism,gene,sequence,length
P12345,Example Kinase,Protein kinase,Homo sapiens,GENE1,MKVL...,450
```

Ingest:

```bash
python -m graphrag.cli ingest targets --input data/graphrag/targets.csv
```

### 3. Ingest Ligands

Prepare `data/graphrag/ligands.csv`:

```csv
ligand_id,name,smiles,inchi_key,scaffold_id,logp,tpsa,mw,source
CHEMBL123,Imatinib,CC1=C...,KTUFNOR...,scaffold_001,2.5,86.0,493.6,ChEMBL
```

Ingest:

```bash
python -m graphrag.cli ingest ligands --input data/graphrag/ligands.csv
```

### 4. Ingest Experimental Structures

Prepare `data/graphrag/structures.csv`:

```csv
model_id,uniprot_id,ligand_id,pdb_path,method,confidence,created_at
pdb_1abc,P12345,CHEMBL123,data/structures/1abc.pdb,xray,1.0,2024-01-01
```

Ingest:

```bash
python -m graphrag.cli ingest structures --input data/graphrag/structures.csv
```

### 5. Ingest WT Complex from AlphaFold / Diffusion

**Step A**: Save predicted complex to:
```
data/graphrag/complex_models/af_model_001.pdb
```

**Step B**: Compute interface contacts and prepare `data/graphrag/contacts.csv`:

```csv
uniprot_id,chain,i,aa_i,j,aa_j,w,dist
P12345,A,45,LEU,120,ASP,1.0,3.8
P12345,A,46,VAL,120,ASP,1.0,4.2
```

**Step C**: Ingest:

```bash
python -m graphrag.cli ingest contacts --input data/graphrag/contacts.csv
```

> 💡 Your numbering is UniProt-consistent, so residue IDs are stored as `(uniprot_id, uniprot_pos)`.

---

## GraphRAG Retrieval + LLM

### 1. Retrieve Subgraph

```bash
python -m graphrag.cli retrieve \
  --query "leukemia target CD19 best antibodies and evidence" \
  --k 50 \
  --out data/graphrag/retrieval_snapshot.json
```

### 2. Summarize and Rank Candidates

```bash
python -m graphrag.cli answer \
  --snapshot data/graphrag/retrieval_snapshot.json \
  --prompt packages/graphrag/prompts/propose_candidates.md \
  --out data/graphrag/candidates_ranked.json
```

**Output** (`candidates_ranked.json`):
```json
{
  "candidates": [
    {
      "type": "antibody",
      "id": "blinatumomab",
      "name": "Blinatumomab",
      "rationale": "FDA-approved bispecific T-cell engager for CD19+ B-cell ALL...",
      "risks": ["Cytokine release syndrome", "Neurotoxicity"]
    }
  ],
  "questions_to_answer": [
    "What is the epitope specificity?",
    "What is the binding affinity range?"
  ]
}
```

### 3. Produce Design Constraints

```bash
python -m graphrag.cli answer \
  --snapshot data/graphrag/retrieval_snapshot.json \
  --prompt packages/graphrag/prompts/propose_constraints.md \
  --out data/graphrag/constraints.json
```

**Output** (`constraints.json`):
```json
{
  "docking_constraints": {
    "must_contact_residues": ["P12345:45", "P12345:120", "P12345:178"],
    "avoid_residues": ["P12345:89"],
    "notes": "Conserved hinge region; mutation hotspot at 89",
    "box_center_hint": "Residue 120",
    "box_size_hint": "20Å x 20Å x 20Å"
  }
}
```

---

## Similarity Search (No Docking Required)

### Goal

Find known complexes **most similar** to your WT predicted interface.

### Prerequisites

Your WT `ComplexModel` must have:
- ✅ Pocket/epitope residue sets
- ✅ Target protein UniProt ID
- ✅ Ligand scaffold / antibody metadata (if available)

### Similarity Metrics

Retrieve based on:
1. **Pocket residue overlap** (Jaccard similarity)
2. **Target family overlap** (protein family annotation)
3. **Ligand scaffold similarity** (chemical fingerprint)
4. **Epitope region overlap** (for antibodies)

### Example Query

```bash
python -m graphrag.cli retrieve-similar \
  --model-id af_model_001 \
  --top-k 10 \
  --out data/graphrag/similar_complexes.json
```

**Output**:
```json
[
  {
    "model_id": "pdb_3k5v",
    "similarity_score": 0.85,
    "overlap_residues": ["45", "120", "178", "180"],
    "target_family": "Protein kinase",
    "ligand_scaffold": "scaffold_001"
  }
]
```

---

## Pipeline: `propose_then_score`

### Overview

End-to-end pipeline connecting GraphRAG to downstream tools:

1. **GraphRAG**: Propose candidates + constraints
2. **(Optional)**: Run external geometry generation (AF/diffusion/docking)
3. **Writeback to Neo4j**:
   - `ComplexModel`
   - `InterfaceResidues`
   - `Pocket` / `Epitope`
   - Confidence metrics
4. **Downstream**:
   - **ResIntNet**: Allostery / hotspot ranking
   - **SeqML**: Mutation proposals + fitness prediction

### Usage

```bash
python -m graphrag.pipelines.propose_then_score \
  --uniprot-id P12345 \
  --pocket-id P12345:pk1 \
  --engine vina_stub \
  --out data/graphrag/run_001/
```

### Output Structure

```
data/graphrag/run_001/
├── proposal.json          # LLM-generated candidate list
├── scored.json            # Candidate scores (stub or real)
├── constraints.json       # Design constraints
└── retrieval_snapshot.json # Subgraph used for decision
```

---

## Compute Constraints

### If You Cannot Run AF/Diffusion Locally

**Workaround**:
1. Generate predicted complexes elsewhere (cloud / HPC)
2. Download PDB/mmCIF files
3. Ingest interface residues + confidence scores

**GraphRAG can still**:
- ✅ Retrieve similar candidates
- ✅ Rank by evidence / toxicity / similarity
- ✅ Produce next-round constraints

---

## Environment Variables Reference

### Neo4j

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Database username |
| `NEO4J_PASSWORD` | `password` | Database password |
| `NEO4J_DATABASE` | `neo4j` | Target database name |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `hf_local` | Provider: `hf_local`, `openai`, etc. |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model ID |
| `LLM_DEVICE` | `auto` | Device: `auto`, `cuda`, `cpu` |
| `LLM_MAX_NEW_TOKENS` | `512` | Max generation length |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |

---

## Advanced: Custom Cypher Queries

### Example: Find All Ligands Binding Target with IC50 < 100nM

```cypher
MATCH (l:Ligand)-[b:BINDS]->(p:Protein {uniprot_id: 'P12345'})
WHERE b.metric = 'IC50' AND b.value < 100 AND b.units = 'nM'
RETURN l.ligand_id, l.name, b.value
ORDER BY b.value ASC
```

### Example: Find Antibodies Without Toxicity Flags

```cypher
MATCH (ab:Antibody)-[:TARGETS]->(p:Protein {uniprot_id: 'P12345'})
WHERE NOT (ab)-[:HAS_TOXICITY]->(:ToxicityEvent)
RETURN ab.antibody_id, ab.name
```

---

## Schema Reference

### Constraints Applied

```cypher
CREATE CONSTRAINT protein_uniprot IF NOT EXISTS
FOR (p:Protein) REQUIRE p.uniprot_id IS UNIQUE;

CREATE CONSTRAINT ligand_id IF NOT EXISTS
FOR (l:Ligand) REQUIRE l.ligand_id IS UNIQUE;

CREATE CONSTRAINT peptide_id IF NOT EXISTS
FOR (p:Peptide) REQUIRE p.peptide_id IS UNIQUE;

CREATE CONSTRAINT pocket_id IF NOT EXISTS
FOR (p:Pocket) REQUIRE p.pocket_id IS UNIQUE;

CREATE CONSTRAINT assay_id IF NOT EXISTS
FOR (a:AssayResult) REQUIRE a.assay_id IS UNIQUE;

CREATE CONSTRAINT tox_id IF NOT EXISTS
FOR (t:ToxicityEvent) REQUIRE t.tox_id IS UNIQUE;

CREATE CONSTRAINT paper_doi IF NOT EXISTS
FOR (p:Paper) REQUIRE p.doi IS UNIQUE;

CREATE CONSTRAINT complex_id IF NOT EXISTS
FOR (c:ComplexModel) REQUIRE c.model_id IS UNIQUE;
```

### Indexes

```cypher
CREATE INDEX protein_family IF NOT EXISTS 
FOR (p:Protein) ON (p.family);

CREATE INDEX ligand_scaffold IF NOT EXISTS 
FOR (l:Ligand) ON (l.scaffold_id);

CREATE INDEX paper_year IF NOT EXISTS 
FOR (p:Paper) ON (p.year);
```

---

## Troubleshooting

### Neo4j Connection Issues

```bash
# Check Neo4j is running
docker ps | grep neo4j

# Check logs
docker logs protein_graphrag_neo4j

# Test connection
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); driver.verify_connectivity(); print('✅ Connected')"
```

### LLM Memory Issues

If running out of VRAM:

```bash
# Use smaller model
export LLM_MODEL="Qwen/Qwen2.5-3B-Instruct"

# Use CPU
export LLM_DEVICE="cpu"

# Use 8-bit quantization (requires bitsandbytes)
export LLM_LOAD_IN_8BIT="true"
```

### Empty Retrieval Results

```bash
# Check data was ingested
docker exec -it protein_graphrag_neo4j cypher-shell -u neo4j -p password

# In cypher-shell:
MATCH (n) RETURN labels(n), count(n);
```

---

## Performance Tips

### 1. Index Critical Properties

Add indexes for frequently queried properties:

```cypher
CREATE INDEX ligand_smiles IF NOT EXISTS 
FOR (l:Ligand) ON (l.smiles);

CREATE INDEX protein_gene IF NOT EXISTS 
FOR (p:Protein) ON (p.gene);
```

### 2. Batch Ingestion

For large datasets (>10K rows), use batch processing:

```python
from graphrag.db import Neo4jClient
from graphrag.config import AppConfig

cfg = AppConfig()
db = Neo4jClient(cfg.neo4j)

# Batch write
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    # Process batch...
```

### 3. Neo4j Memory Tuning

Edit `docker-compose.neo4j.yml`:

```yaml
environment:
  - NEO4J_server_memory_pagecache_size=2G
  - NEO4J_server_memory_heap_initial__size=2G
  - NEO4J_server_memory_heap_max__size=4G
```

---

## Contributing

### Adding New Node Types

1. Update `schema/nodes.yaml`
2. Add constraint in `schema/constraints.cql`
3. Create ingest script in `src/graphrag/ingest/`
4. Update retrieval queries in `src/graphrag/retrieval/cypher.py`

### Adding New LLM Providers

1. Create provider class in `src/graphrag/llm/`
2. Inherit from `LLM` base class
3. Register in `llm/loader.py`

---

## License

This package follows the repository license.

**External datasets/tools** keep their own terms:
- Check **Open Targets**, **SAbDab**, **PDBbind**, **ChEMBL** licenses before redistribution/commercial use.
- AlphaFold predictions: See [AlphaFold license](https://github.com/deepmind/alphafold)

---

## Citation

If you use this package in research, please cite:

```bibtex
@software{graphrag_protein_engineering,
  title={GraphRAG Package for Protein Engineering},
  author={Your Lab/Organization},
  year={2024},
  url={https://github.com/yourorg/protein-engineering}
}
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourorg/protein-engineering/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourorg/protein-engineering/discussions)
- **Documentation**: [Full Docs](https://yourorg.github.io/protein-engineering/graphrag)

---

**Last Updated**: December 2024  
**Package Version**: 0.1.0
