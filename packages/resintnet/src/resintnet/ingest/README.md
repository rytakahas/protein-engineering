# ResIntNet Ingestion Adapters

Adapter-based ingestion pipeline to normalize distal/allosteric mutation datasets.

## Quickstart
D3Distal:
```
python packages/resintnet/scripts/ingest_mutations.py   --source d3distal   --in data/ext/D3DistalMutation_export.csv   --out-mutations data/labels/mutations.csv   --out-residues  data/labels/residues.csv
```

Generic CSV with YAML:
```
python packages/resintnet/scripts/ingest_mutations.py   --source generic   --in data/ext/paperX_table1.csv   --mapping-yaml configs/mappings/generic_example.yaml   --out-mutations data/labels/mutations_lit.csv   --out-residues  data/labels/residues_lit.csv
```
