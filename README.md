## Protein Engineering Monorepo — ResContact → GraphRAG → ResIntNet → SeqML

End-to-end, modular pipeline for **protein & enzyme engineering**:

from **sequence** (± structure) and experimental datasets →  
**contacts / priors** (fast, low-cost) →  
**GraphRAG-driven target / ligand / evidence reasoning** →  
**residue-interaction graphs & allosteric hotspot ranking** (distal mutation candidates) →  
**sequence-level efficacy / fitness modeling** (fine-tuned LMs) →  
**experiment-in-the-loop iteration**.

This repository is organized as a **monorepo** with independent packages and thin orchestration under `pipelines/`.  
Each package is installable & runnable on its own; outputs are cached to disk so stages can be resumed or reused.

---

## Table of Contents

- [Goals](#goals)
- [High-level architecture](#high-level-architecture)
- [Directory layout](#directory-layout)
- [Install](#install)
- [Data & caches](#data--caches)
- [Stage A — ResContact (contacts, priors, features)](#stage-a--rescontact-contacts-priors-features)
- [Stage B — GraphRAG (targets, candidates, evidence)](#stage-b--graphrag-targets-candidates-evidence)
- [Stage C — ResIntNet (graphs, PRS/memory, GNN ranking)](#stage-c--resintnet-graphs-prsmemory-gnn-ranking)
- [Stage D — SeqML (mutant efficacy / fitness modeling)](#stage-d--seqml-mutant-efficacy--fitness-modeling)
- [Pipelines (E2E orchestration)](#pipelines-e2e-orchestration)
- [Configs](#configs)
- [Tips (Mac M1/M2/M3, CUDA), Repro, Docker](#tips-mac-m1m2m3-cuda-repro-docker)
- [Licenses & data](#licenses--data)
- [What's next](#whats-next)

---

## Goals

- **rescontact**  
  Build **lightweight, local** contact / structure priors from sequence using **ESM2 + MSA + template statistics**.  
  This stage provides **quick, low-cost structural sanity checks** and residue–residue interaction estimates.

  > **Note**  
  > When higher-fidelity folding or complex prediction tools are available  
  > (e.g. AlphaFold Server, ColabFold, OpenFold, newer open complex predictors),  
  > ResContact can be **replaced or skipped**.  
  > It remains useful for rapid screening, large batches, and laptop-scale/offline workflows.

- **graphrag**  
  Integrate **targets, ligands/peptides, pockets, residues, assays, toxicity, and literature** into a **knowledge graph** (Neo4j).  
  Use **GraphRAG + LLMs** to retrieve structured evidence, propose **candidate binders**, and generate **constraints** for docking or complex prediction.

- **resintnet**  
  Convert contacts/structures into **residue-interaction graphs**; compute  
  **PRS (Perturbation-Response Scanning)** + **adaptive memory** signals and **GNN** scores to rank **distal allosteric hotspots**.

- **seqml**  
  Fine-tune sequence models (**T5 / ESM-style PLMs**) for **mutant efficacy / fitness prediction** around proposed residues; iterate with experimental data.

**Design principle**

- **Geometry** comes from **structure / docking / complex prediction tools**.  
- **Decision-making, evidence integration, and prioritization** come from **GraphRAG + LLMs**.

---

## High-level architecture

```
(sequence + templates + MSAs)
                │
                ▼
      ┌────────────────────────────┐
      │   Stage A: ResContact      │
      │ • fast contact priors      │
      │ • cheap structure checks   │
      └─────────┬──────────────────┘
                │
                ▼
      ┌────────────────────────────┐
      │  Stage B: GraphRAG (KG)    │  ← Neo4j + LLM
      │ • targets & families       │
      │ • ligands / peptides       │
      │ • assays & toxicity        │
      │ • literature evidence      │
      │ → candidate proposals      │
      │ → docking constraints      │
      └─────────┬──────────────────┘
                │
        (quick docking / complex prediction)
                │
                ▼
      ┌────────────────────────────┐
      │   Stage C: ResIntNet       │
      │ • residue graph            │
      │ • PRS + adaptive memory    │
      │ • GNN hotspot ranking      │
      └─────────┬──────────────────┘
                │
                ▼
      ┌────────────────────────────┐
      │   Stage D: SeqML           │
      │ • mutant generation        │
      │ • PLM fine-tuning          │
      │ • fitness prediction       │
      └─────────┬──────────────────┘
                │
                ▼
          (experiments / databases)
                │
                └── write back to GraphRAG
```

---

## Directory layout

```
.
├── README.md
├── archive/
│   └── msa_utils/
│       ├── Dockerfile
│       ├── README.md
│       └── run_deepmsa.sh
├── configs/
│   ├── mappings/
│   │   └── generic_example.yaml       # example for mapping/query metadata
│   └── pipeline.example.yaml          # E2E config template
├── docker/                             # (future) base images / compose
├── packages/
│   ├── rescontact/                     # Stage A
│   │   ├── README.md
│   │   ├── notebooks/
│   │   │   ├── res_contact_workflow.ipynb
│   │   │   ├── res_contact_workflow_opt.ipynb
│   │   │   ├── res_contact_workflow_opt2.ipynb
│   │   │   └── res_contact_workflow_opt_msa.ipynb
│   │   ├── pyproject.toml
│   │   ├── scripts/
│   │   │   ├── retrieve_homologs.py
│   │   │   ├── run_msa_batch.py
│   │   │   ├── build_template_priors.py
│   │   │   ├── build_msa_features.py
│   │   │   ├── embed_esm2.py
│   │   │   ├── concat_esm2_msa.py
│   │   │   ├── fuse_priors.py
│   │   │   ├── train.py
│   │   │   ├── eval.py
│   │   │   └── psi_report.py
│   │   └── src/rescontact/
│   │       ├── api/ features/ io/ datasets/ models/ training/ utils/
│   ├── graphrag/                       # Stage B (NEW)
│   │   ├── README.md
│   │   ├── schema/                     # node/edge definitions
│   │   ├── ingest/                     # targets, ligands, assays, ResIntNet outputs
│   │   ├── retrieval/                  # subgraph builders
│   │   ├── prompts/                    # LLM prompts
│   │   └── pipelines/
│   │       └── propose_then_score.py
│   ├── resintnet/                      # Stage C
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   ├── scripts/
│   │   │   ├── ingest_mutations.py
│   │   │   └── rank_mutations.py
│   │   └── src/resintnet/
│   │       ├── graph.py prs.py rank.py
│   │       ├── models/
│   │       │   └── sage.py
│   │       └── ingest/
│   │           ├── base.py utils.py
│   │           └── adapters/
│   │               ├── d3distal.py
│   │               └── generic_csv.py
│   └── seqml/                          # Stage D
│       ├── README.md
│       ├── pyproject.toml
│       ├── scripts/
│       │   ├── prepare_mutants.py
│       │   └── train.py
│       └── src/seqml/
│           ├── mutgen.py
│           └── train_t5_lora.py
└── pipelines/
    ├── README.md
    ├── e2e_propose_mutations.py
    ├── train_rescontact.py
    ├── train_resintnet.py
    └── train_seqml.py
```

---

## Install

Use per-package editable installs (keeps deps isolated). Recommended Python 3.11.

```bash
# From repo root
python -m venv .venv && source .venv/bin/activate

# Stage A
pip install -e packages/rescontact

# Stage B
pip install -e packages/graphrag

# Stage C
pip install -e packages/resintnet

# Stage D
pip install -e packages/seqml
```

---

## Data & caches

All stages write to local caches (NPZ/NPY/Parquet/CSV), enabling resume and reuse:

- `data/fasta/*.fa` — query sequences
- `data/templates/priors/*.npz` — ResContact template priors
- `data/emb/esm2_*/*.npy` — ESM2 embeddings
- `data/msas/*.a3m`, `data/msa_features/*.npz` — MSAs + summaries
- `data/graphrag/*` — KG exports, candidate proposals, retrieval snapshots
- `data/resintnet/*` — graphs, PRS/memory scores, rankings
- `data/seqml/*` — mutants, datasets, checkpoints

---

## Stage A — ResContact (contacts, priors, features)

### Purpose

Provide cheap, local contact and structure estimates directly from sequence.

ResContact is designed as:

- a **first-pass structural filter**
- a **fallback** when full folding is unavailable
- a **scalable option** for many sequences

If high-quality folding or complex prediction is available, ResContact outputs can be replaced by:

- AlphaFold Server / ColabFold
- OpenFold
- newer open complex predictors

without changing downstream stages.

---

## Stage B — GraphRAG (Knowledge Graph: targets → candidates → evidence)

### Purpose
Stage B is the **decision + memory layer**. It connects:
- **structural signals** (contacts, pockets, residue neighborhoods)
with
- **biological/chemical/experimental knowledge** (targets, families, ligands, assays, toxicity, papers),

so we can **propose candidates**, **justify them with evidence**, and **keep project memory across iterations**.

> Key idea: the KG is built **offline** (ingest + normalize + type with an ontology/schema),
> and GraphRAG runs **online** on **small, retrieved subgraphs** (not the entire graph).

---

### What GraphRAG does (and does not do)
**Does**
- Retrieve related **targets / families / motifs / resistance sites**
- Propose **ligands / peptides / binder scaffolds**
- Generate **constraints** for docking / complex prediction (e.g., pocket residues, interaction hints)
- Produce **evidence-backed explanations** (“why this residue / why this ligand”)
- Track **history** (what we tried, what worked, what failed)

**Does NOT**
- Compute binding geometry
- Replace folding/docking engines
- Provide accurate binding free energies (that belongs to physics + docking + experiments)

---

### KG modeling here: same layers as relational, different artifacts
You still do **conceptual → logical → physical** modeling, just “graph-shaped”:

**Conceptual (business/science view)**
- Entities + relationships you care about (Protein–Residue–Pocket–Ligand–Assay–Paper…)
- Query patterns (the questions you need to answer)

**Logical (graph schema / ontology)**
- Node labels, relationship types, property conventions, cardinalities
- Normalization rules (canonical IDs, synonyms, provenance)
- Which relations are “safe to traverse” for retrieval (important for performance + relevance)

**Physical (performance + scale)**
- Uniqueness constraints on canonical IDs (e.g., UniProt/PDB/ChEMBL/DOI)
- Indexes for fast seeding (name/synonym lookups, pocket IDs, assay IDs)
- Optional: full-text index for paper metadata; vector index for embeddings
- Optional: precomputed adjacency summaries for common 1–2 hop expansions

---

### Retrieval strategy: BFS for context, DFS for explanation (both bounded)
A common misunderstanding is “GraphRAG = graph traversal.”  
In practice: **GraphRAG = indexed retrieval + constrained traversal + ranking + trimming**.

#### 1) Seed (index-first, not traversal-first)
Start from **anchored nodes** identified from the user question:
- exact IDs (UniProt, PDB, ChEMBL, DOI),
- or synonym-resolved names (protein aliases, ligand names).

This step must be O(log N) style lookups via indexes/constraints.

#### 2) Bounded BFS (shallow neighborhood = fast context)
Use **k-hop BFS** (typically k=2 or 3) to collect a **local evidence neighborhood**:
- “What’s directly connected to this protein/pocket/residue?”
- “Which assays/papers mention this ligand + target?”
This maximizes **coverage** of relevant context quickly.

#### 3) Guided DFS / path search (thin chains = good narratives)
Then run a **guided DFS** (or constrained path search) *inside that retrieved subgraph* to find:
- short, high-signal paths that explain *why* something is proposed:
  - `Protein → Pocket → Ligand → AssayResult → Paper`
  - `Residue → ALLOSTERIC_PATH → Pocket → Ligand`

DFS is not “deep crawl everything.” It is:
- bounded depth,
- limited relationship types,
- optionally weighted (edge confidence / recency / evidence strength).

#### 4) Rank + trim (make it LLM-sized)
Finally:
- rank nodes/edges (evidence strength, confidence, recency, diversity),
- keep **top-N** nodes/edges and **top-K** paths,
- generate the prompt context from this **compact subgraph**.

---

### Minimal starter schema (property graph view)
**Nodes**
- `Protein`, `Residue`, `Pocket`
- `Ligand`, `Peptide`
- `ComplexModel`
- `AssayResult`
- `ToxicityEvent`
- `Paper`

**Edges**
- `HAS_RESIDUE`, `COMPOSED_OF`
- `BINDS`, `PREDICTED_BINDING`
- `ALLOSTERIC_PATH`
- `HAS_TOXICITY`
- `MENTIONED_IN`, `MEASURES`

**Provenance (recommended)**
Attach `source`, `source_url`, `timestamp`, `confidence` to nodes/edges
so every retrieved claim can be traced.

---

### Example query patterns this stage should answer
- “Given **Protein X**, what **pockets** are plausible and which **ligands** are supported by assays/papers?”
- “Why is **Residue r123** a good mutation target? Show evidence chain(s).”
- “Find ligand candidates for **family Y** with low toxicity signals and relevant assays.”

---

### Output of Stage B (what gets passed downstream)
- Candidate set: `(pocket/residue targets, ligand/peptide candidates)`
- Evidence pack: compact subgraph + top explanatory paths + citations/provenance
- Constraints for Stage C/D: residues/pockets to prioritize + rationale


---

## Stage C — ResIntNet (graphs, PRS/memory, GNN ranking)

### Input

- Contacts/structures (from ResContact or folding tools)
- Per-residue embeddings
- Optional docking-derived contacts

### Output

- Ranked distal allosteric hotspots with interpretable scores

### Core ideas

- Residue-interaction graph
- PRS (ENM-style allosteric proxy)
- Adaptive memory conductance C
- GNN-based pattern learning
- Ensemble scoring (physics + learning)

---

## Stage D — SeqML (mutant efficacy / fitness modeling)

### Purpose

Learn sequence → phenotype mappings around proposed hotspots.

- Generate focused mutant neighborhoods
- Fine-tune PLMs with experimental data
- Predict efficacy / stability / activity
- Feed results back into GraphRAG

---

## Pipelines (E2E orchestration)

```bash
python pipelines/e2e_propose_mutations.py --config configs/pipeline.example.yaml
```

This performs:

1. ResContact fast structure priors
2. GraphRAG retrieval + candidate proposal
3. Docking / complex quick evaluation
4. ResIntNet hotspot ranking
5. SeqML mutant scoring

---

## Configs

Use a single YAML to configure the end-to-end run:

- `configs/pipeline.example.yaml`

### LLM choice (important clarification)

This repo does not rely on protein-specific LLMs (ProtBERT, ProtT5) for reasoning.

- **ProtBERT / ProtT5 / ESM** → used in SeqML as encoders
- **General open-weight LLMs** → used in GraphRAG for:
  - entity extraction
  - evidence summarization
  - candidate prioritization

---

## Tips (Mac M1/M2/M3, CUDA), Repro, Docker

- Apple Silicon works well with PyTorch MPS for embeddings and small models
- Docking + GraphRAG run well on CPU
- Heavy folding can be offloaded (Colab / server)
- Docker images (planned) will pin toolchains

---

## Licenses & data

- **Code**: repo license (TBD)
- **External datasets and tools** retain their own licenses
- Verify terms before redistribution or commercial use

---

## What's next

- Add Neo4j schema + migrations
- Add docking adapters (Vina / smina / DiffDock)
- Add CI smoke tests
- Add example notebooks:
  - Target → ligand → hotspot → mutant loop
