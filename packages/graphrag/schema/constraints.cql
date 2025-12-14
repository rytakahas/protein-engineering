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

CREATE INDEX protein_family IF NOT EXISTS FOR (p:Protein) ON (p.family);
CREATE INDEX ligand_scaffold IF NOT EXISTS FOR (l:Ligand) ON (l.scaffold_id);
CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year);
