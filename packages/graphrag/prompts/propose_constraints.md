
# Task: Propose design / docking / complex constraints from a retrieved subgraph

Question: {{question}}
Subgraph JSON: {{snapshot_json}}

## Instructions
- Use ONLY information from the subgraph.
- You are NOT computing 3D poses. You are proposing constraints / residue lists / avoid lists.

## Output (STRICT JSON only)

{
  "must_contact_residues": ["UNIPROT:POS", "..."],
  "avoid_residues": ["UNIPROT:POS", "..."],
  "avoid_chemotypes": ["string", "..."],
  "notes": ["short notes"],
  "confidence": 0.0
}

No markdown. No extra keys.
