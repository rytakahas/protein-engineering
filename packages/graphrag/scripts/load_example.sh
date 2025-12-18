#!/usr/bin/env bash
set -euo pipefail

set -a
if [ -f .env ]; then source .env; fi
set +a

python -m graphrag.cli schema apply --cql packages/graphrag/schema/constraints.cql

python -m graphrag.cli ingest targets    --input data/graphrag/targets.csv
python -m graphrag.cli ingest ligands    --input data/graphrag/ligands.csv
python -m graphrag.cli ingest assays     --input data/graphrag/assays.csv
python -m graphrag.cli ingest toxicity   --input data/graphrag/toxicity.csv
python -m graphrag.cli ingest structures --input data/graphrag/structures.csv
python -m graphrag.cli ingest contacts   --input data/graphrag/contacts.csv

python -m graphrag.cli stats
