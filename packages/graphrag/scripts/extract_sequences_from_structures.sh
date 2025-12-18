#!/usr/bin/env bash
set -euo pipefail

set -a
if [ -f .env ]; then source .env; fi
set +a

python -m graphrag.structure.extract_sequences \
  --structures data/graphrag/structures.csv \
  --out data/graphrag/structure_sequences.json

echo "✅ wrote data/graphrag/structure_sequences.json"
