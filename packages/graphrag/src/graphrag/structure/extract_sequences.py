from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .pdb_utils import extract_chain_sequences

def _read_structures_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract chain sequences from PDBs listed in structures.csv")
    ap.add_argument("--structures", required=True, help="Path to structures.csv (needs model_id + pdb_path)")
    ap.add_argument("--out", required=True, help="Output JSON file")
    args = ap.parse_args()

    rows = _read_structures_csv(args.structures)
    out: Dict[str, Any] = {"structures": []}

    for r in rows:
        pdb_path = r.get("pdb_path") or r.get("path") or ""
        model_id = r.get("model_id") or r.get("structure_id") or r.get("pdb_id") or ""
        if not pdb_path or not model_id:
            continue

        p = Path(pdb_path)
        if not p.exists():
            out["structures"].append({"model_id": model_id, "pdb_path": pdb_path, "error": "missing_file"})
            continue

        try:
            chains = extract_chain_sequences(pdb_path)
            out["structures"].append(
                {
                    "model_id": model_id,
                    "uniprot_id": r.get("uniprot_id") or "",
                    "pdb_path": pdb_path,
                    "chains": {cid: {"sequence": c["sequence"], "n_res": len(c["sequence"])} for cid, c in chains.items()},
                }
            )
        except Exception as e:
            out["structures"].append({"model_id": model_id, "pdb_path": pdb_path, "error": str(e)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ wrote {args.out}")

if __name__ == "__main__":
    main()
