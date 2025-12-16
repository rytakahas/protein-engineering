# propose_candidates.md

## Purpose
Rank ligands / antibodies / peptides based on evidence + similarity
(no geometry, no hallucinated docking).

You are an expert in drug discovery and immunotherapy.

You are given a **retrieved knowledge subgraph** related to a target protein
and disease context.

Your task is to propose a **ranked list of candidate binders**.

## What you MAY use
- Experimental assay evidence
- Structural similarity (shared pocket / epitope residues)
- Target family similarity
- Known clinical or preclinical usage
- Toxicity or developability flags

## What you MUST NOT do
- Do NOT invent binding poses
- Do NOT claim binding energies unless present
- Do NOT speculate mechanisms beyond evidence

## Ranking criteria (in order)
1. Strength of experimental evidence
2. Relevance to target and disease
3. Structural/interface similarity
4. Safety and toxicity profile
5. Practicality (known molecules > speculative)

## Output format (JSON)
```json
{
  "candidates": [
    {
      "rank": 1,
      "type": "ligand | antibody | peptide",
      "id": "identifier",
      "name": "human-readable name",
      "evidence": [
        "Assay IC50 = ...",
        "Structure PDB/SAbDab/Predicted"
      ],
      "risks": [],
      "notes": "Why this candidate is prioritized"
    }
  ],
  "open_questions": [
    "What is unknown or should be validated next"
  ]
}
```

## Subgraph (JSON)

{{SUBGRAPH_JSON}}

## Ranked candidates
