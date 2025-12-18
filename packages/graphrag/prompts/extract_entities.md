
Extract the key entities from the question.

Question: {{question}}

Return STRICT JSON only:

{
  "targets": ["..."],
  "diseases": ["..."],
  "candidate_types": ["ligand","antibody","peptide"],
  "keywords": ["..."]
}
