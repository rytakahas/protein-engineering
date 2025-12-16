# propose_constraints.md

## Purpose  
Generate **design / docking / diffusion constraints**  
from evidence and WT predicted interfaces.

You are assisting a structure-based design workflow.

You are given:
- a target protein
- known or predicted complexes (including AlphaFold / diffusion WT models)
- residue-level interface or contact information

Your task is to propose **design constraints** for the next modeling round.

## Allowed constraints
- Must-contact residues
- Epitope or pocket residue sets
- Avoid residues (toxicity, resistance, instability)
- Soft spatial hints (NOT exact coordinates)

## Forbidden
- Do NOT invent precise 3D coordinates
- Do NOT estimate binding energies
- Do NOT claim pose accuracy

## Output format (JSON)
```json
{
  "interface_constraints": {
    "must_contact_residues": ["UNIPROT:POS"],
    "preferred_region": "N-terminal pocket / epitope region",
    "avoid_residues": ["UNIPROT:POS"],
    "confidence_level": "high | medium | low",
    "notes": "Why these constraints were chosen"
  },
  "modeling_recommendations": {
    "suggested_methods": ["AlphaFold-multimer", "diffusion"],
    "reference_templates": ["PDB_ID or model_id"]
  }
}
```

## Subgraph (JSON)

{{SUBGRAPH_JSON}}

## Constraints
