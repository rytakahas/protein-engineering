from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common alternative residues
    "MSE": "M",  # selenomethionine
}

@dataclass(frozen=True)
class ResidueRec:
    chain: str
    resseq: int
    icode: str
    resname: str
    aa1: str
    ca: Optional[Tuple[float, float, float]]

def _safe_import_biopython():
    try:
        from Bio.PDB import PDBParser  # type: ignore
        return PDBParser
    except Exception as e:
        raise ImportError(
            "Biopython is required for PDB parsing.\n"
            "Install with: pip install biopython\n"
            f"Original error: {e}"
        )

def _res_aa1(resname: str) -> str:
    return _AA3_TO_1.get((resname or "").upper(), "X")

def extract_chain_sequences(pdb_path: str | Path) -> Dict[str, Dict[str, object]]:
    """
    Parse a PDB and return per-chain sequences + residue table.

    Returns:
      {
        "A": {
          "sequence": "MKT...",
          "residues": [ {chain, resseq, icode, resname, aa1}, ... ]
        },
        ...
      }
    """
    pdb_path = str(pdb_path)
    PDBParser = _safe_import_biopython()
    parser = PDBParser(QUIET=True)

    structure = parser.get_structure("model", pdb_path)
    model = next(structure.get_models())

    out: Dict[str, Dict[str, object]] = {}
    for chain in model.get_chains():
        chain_id = chain.id
        residues: List[ResidueRec] = []
        seq_chars: List[str] = []

        for res in chain.get_residues():
            hetflag, resseq, icode = res.id
            if hetflag == "W":  # water
                continue

            resname = (res.resname or "").upper()
            aa1 = _res_aa1(resname)

            # only AA-like residues contribute to sequence
            if aa1 == "X":
                continue

            ca = None
            if "CA" in res:
                atom = res["CA"]
                ca = (float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2]))

            residues.append(
                ResidueRec(chain=chain_id, resseq=int(resseq), icode=str(icode or ""), resname=resname, aa1=aa1, ca=ca)
            )
            seq_chars.append(aa1)

        out[chain_id] = {
            "sequence": "".join(seq_chars),
            "residues": [
                {"chain": r.chain, "resseq": r.resseq, "icode": r.icode, "resname": r.resname, "aa1": r.aa1}
                for r in residues
            ],
        }

    return out

def extract_ca_coords(pdb_path: str | Path) -> List[ResidueRec]:
    """Return a flat list of AA residues that have CA coordinates."""
    pdb_path = str(pdb_path)
    PDBParser = _safe_import_biopython()
    parser = PDBParser(QUIET=True)

    structure = parser.get_structure("model", pdb_path)
    model = next(structure.get_models())

    recs: List[ResidueRec] = []
    for chain in model.get_chains():
        for res in chain.get_residues():
            hetflag, resseq, icode = res.id
            if hetflag == "W":
                continue

            resname = (res.resname or "").upper()
            aa1 = _res_aa1(resname)
            if aa1 == "X":
                continue
            if "CA" not in res:
                continue

            atom = res["CA"]
            ca = (float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2]))
            recs.append(
                ResidueRec(chain=chain.id, resseq=int(resseq), icode=str(icode or ""), resname=resname, aa1=aa1, ca=ca)
            )
    return recs

def euclidean(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
