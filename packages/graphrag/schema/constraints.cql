
CREATE CONSTRAINT protein_uniprot IF NOT EXISTS
FOR (p:Protein) REQUIRE p.uniprot_id IS UNIQUE;

CREATE CONSTRAINT disease_id IF NOT EXISTS
FOR (d:Disease) REQUIRE d.disease_id IS UNIQUE;

CREATE CONSTRAINT ligand_id IF NOT EXISTS
FOR (l:Ligand) REQUIRE l.ligand_id IS UNIQUE;

CREATE CONSTRAINT antibody_id IF NOT EXISTS
FOR (a:Antibody) REQUIRE a.antibody_id IS UNIQUE;

CREATE CONSTRAINT peptide_id IF NOT EXISTS
FOR (p:Peptide) REQUIRE p.peptide_id IS UNIQUE;

CREATE CONSTRAINT structure_id IF NOT EXISTS
FOR (s:Structure) REQUIRE s.structure_id IS UNIQUE;

CREATE CONSTRAINT complex_model_id IF NOT EXISTS
FOR (c:ComplexModel) REQUIRE c.model_id IS UNIQUE;

CREATE CONSTRAINT assay_id IF NOT EXISTS
FOR (a:AssayResult) REQUIRE a.assay_id IS UNIQUE;

CREATE CONSTRAINT tox_id IF NOT EXISTS
FOR (t:ToxicityEvent) REQUIRE t.tox_id IS UNIQUE;

CREATE CONSTRAINT paper_doi IF NOT EXISTS
FOR (p:Paper) REQUIRE p.doi IS UNIQUE;

CREATE INDEX protein_gene IF NOT EXISTS
FOR (p:Protein) ON (p.gene);

CREATE INDEX protein_name IF NOT EXISTS
FOR (p:Protein) ON (p.name);

CREATE INDEX ligand_name IF NOT EXISTS
FOR (l:Ligand) ON (l.name);

CREATE INDEX ligand_scaffold IF NOT EXISTS
FOR (l:Ligand) ON (l.scaffold_id);

CREATE INDEX disease_name IF NOT EXISTS
FOR (d:Disease) ON (d.name);

CREATE INDEX structure_method IF NOT EXISTS
FOR (s:Structure) ON (s.method);

CREATE INDEX complex_method IF NOT EXISTS
FOR (c:ComplexModel) ON (c.method);

CREATE INDEX assay_metric IF NOT EXISTS
FOR (a:AssayResult) ON (a.metric);

CREATE INDEX tox_severity IF NOT EXISTS
FOR (t:ToxicityEvent) ON (t.severity);

CREATE INDEX paper_year IF NOT EXISTS
FOR (p:Paper) ON (p.year);

CREATE INDEX residue_uid IF NOT EXISTS
FOR (r:Residue) ON (r.residue_uid);
