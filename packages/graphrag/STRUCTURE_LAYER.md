# Structure layer (PDB parsing + contact generation)

This zip adds a minimal "structure understanding + mapping" layer:

- Parse PDBs listed in `data/graphrag/structures.csv`
- Extract per-chain sequences (sanity checks)
- Build an intra-protein residue contact network (CA-CA within a cutoff)
- Optionally map chain residue order -> UniProt positions if:
  - `targets.csv` includes a `sequence` for that `uniprot_id`
  - Biopython alignment is installed (`pip install biopython`)

## Install deps

```bash
pip install biopython numpy
# optional speed-up for larger proteins:
pip install scipy
```

## Run

```bash
# optional: extract sequences for inspection
python -m graphrag.structure.extract_sequences \
  --structures data/graphrag/structures.csv \
  --out data/graphrag/structure_sequences.json

# generate contacts from PDBs
python -m graphrag.structure.build_contacts \
  --structures data/graphrag/structures.csv \
  --targets data/graphrag/targets.csv \
  --out data/graphrag/contacts_from_structures.csv \
  --distance 8.0 \
  --contact-type intra_protein

# ingest into Neo4j
python -m graphrag.cli ingest contacts --input data/graphrag/contacts_from_structures.csv
```

## What this does not do yet

- pocket detection
- ligand/protein interface contacts (heteroatoms)
- SIFTS mapping (PDB <-> UniProt residue numbering)
- extracting binding-site residues from ligands/antibodies/peptides automatically

Those can be added next once your `structures.csv` contains explicit partner IDs
(e.g., ligand_id / antibody_id / peptide_id for each structure).
