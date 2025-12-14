SUBGRAPH_BY_PROTEIN = """
MATCH (p:Protein {uniprot_id:$uniprot_id})
OPTIONAL MATCH (p)-[:HAS_POCKET]->(pk:Pocket)
OPTIONAL MATCH (pk)-[:COMPOSED_OF]->(r:Residue)
OPTIONAL MATCH (l:Ligand)-[b:BINDS]->(p)
OPTIONAL MATCH (l)-[:HAS_TOXICITY]->(t:ToxicityEvent)
OPTIONAL MATCH (a:AssayResult)
  WHERE a.uniprot_id = $uniprot_id AND (a.ligand_id = l.ligand_id OR a.peptide_id IS NOT NULL)
OPTIONAL MATCH (x)-[:MENTIONED_IN]->(paper:Paper)
  WHERE (x:Protein AND x.uniprot_id=$uniprot_id) OR (x:Ligand AND x.ligand_id=l.ligand_id)
RETURN p, collect(distinct pk) as pockets, collect(distinct r) as residues,
       collect(distinct l) as ligands, collect(distinct b) as binds,
       collect(distinct t) as tox, collect(distinct a) as assays,
       collect(distinct paper) as papers
"""
