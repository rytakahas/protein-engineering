
# Task: Rank candidates from a retrieved subgraph

You are given:
- A user question: {{question}}
- A retrieved subgraph in JSON: {{snapshot_json}}

## Instructions
1) Use ONLY information in the subgraph.
2) If the subgraph is empty, say so and ask for ingestion/retrieval.
3) Produce a ranked list of candidates (ligands / antibodies / peptides) relevant to the target.

## Output format (STRICT JSON only)
Return JSON with this schema:

{
  "candidates": [
    {
      "rank": 1,
      "type": "ligand|antibody|peptide",
      "id": "string",
      "name": "string",
      "why": ["bullet evidence statements grounded in subgraph edges/nodes"],
      "risks": ["toxicity / uncertainty / missing evidence"]
    }
  ],
  "missing_info": ["what evidence is missing to be confident"],
  "next_queries": ["2-4 follow-up retrieval queries to run in GraphRAG"]
}

Do not include markdown. Do not include extra keys.
