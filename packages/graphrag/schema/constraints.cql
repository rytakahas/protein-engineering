// -----------------------------
// GraphRAG Neo4j constraints + indexes
// Neo4j 5.x compatible
// -----------------------------

// ---- Uniqueness constraints ----
CREATE CONSTRAINT protein_uniprot IF NOT EXISTS
FOR (p:Protein)
REQUIRE p.uniprot_id IS UNIQUE;

CREATE CONSTRAINT disease_id IF NOT EXISTS
FOR (d:Disease)
REQUIRE d.disease_id IS UNIQUE;

CREATE CONSTRAINT ligand_id IF NOT EXISTS
FOR (l:Ligand)
REQUIRE l.ligand_id IS UNIQUE;

CREATE CONSTRAINT antibody_id IF NOT EXISTS
FOR (a:Antibody)
REQUIRE a.antibody_id IS UNIQUE;

CREATE CONSTRAINT peptide_id IF NOT EXISTS
FOR (p:Peptide)
REQUIRE p.peptide_id IS UNIQUE;

CREATE CONSTRAINT structure_id IF NOT EXISTS
FOR (s:Structure)
REQUIRE s.structure_id IS UNIQUE;

CREATE CONSTRAINT complexmodel_id IF NOT EXISTS
FOR (c:ComplexModel)
REQUIRE c.model_id IS UNIQUE;

CREATE CONSTRAINT assay_id IF NOT EXISTS
FOR (a:AssayResult)
REQUIRE a.assay_id IS UNIQUE;

CREATE CONSTRAINT toxicity_id IF NOT EXISTS
FOR (t:ToxicityEvent)
REQUIRE t.tox_id IS UNIQUE;

CREATE CONSTRAINT paper_id IF NOT EXISTS
FOR (p:Paper)
REQUIRE p.paper_id IS UNIQUE;

CREATE CONSTRAINT residue_uid IF NOT EXISTS
FOR (r:Residue)
REQUIRE r.residue_uid IS UNIQUE;

CREATE CONSTRAINT pocket_id IF NOT EXISTS
FOR (p:Pocket)
REQUIRE p.pocket_id IS UNIQUE;

CREATE CONSTRAINT epitope_id IF NOT EXISTS
FOR (e:Epitope)
REQUIRE e.epitope_id IS UNIQUE;


// ---- Helpful property indexes (not unique) ----
CREATE INDEX protein_gene IF NOT EXISTS
FOR (p:Protein) ON (p.gene);

CREATE INDEX protein_family IF NOT EXISTS
FOR (p:Protein) ON (p.family);

CREATE INDEX ligand_scaffold IF NOT EXISTS
FOR (l:Ligand) ON (l.scaffold_id);

CREATE INDEX ligand_inchikey IF NOT EXISTS
FOR (l:Ligand) ON (l.inchi_key);

CREATE INDEX disease_name IF NOT EXISTS
FOR (d:Disease) ON (d.name);

CREATE INDEX paper_year IF NOT EXISTS
FOR (p:Paper) ON (p.year);

// Optional: if you store a short “fingerprint hash” for pockets/epitopes
CREATE INDEX pocket_fp IF NOT EXISTS
FOR (p:Pocket) ON (p.fingerprint);

CREATE INDEX epitope_fp IF NOT EXISTS
FOR (e:Epitope) ON (e.fingerprint);

