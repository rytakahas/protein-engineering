#!/usr/bin/env bash
set -euo pipefail

set -a
if [ -f .env ]; then source .env; fi
set +a

python -m graphrag.structure.build_contacts \
  --structures data/graphrag/structures.csv \
  --targets data/graphrag/targets.csv \
  --out data/graphrag/contacts_from_structures.csv \
  --distance 8.0 \
  --contact-type intra_protein

echo "✅ wrote data/graphrag/contacts_from_structures.csv"
echo "Now ingest it:"
echo "  python -m graphrag.cli ingest contacts --input data/graphrag/contacts_from_structures.csv"
