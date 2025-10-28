
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder

def _parser_for(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdb":
        return PDBParser(QUIET=True)
    if ext in (".cif", ".mmcif"):
        return MMCIFParser(QUIET=True)
    raise ValueError(f"Unsupported structure format: {path}")

def load_structure(path: Path):
    parser = _parser_for(path)
    struct = parser.get_structure(path.stem, str(path))
    return struct

def extract_seqres_by_chain(struct) -> Dict[str, str]:
    """Return AA sequence per chain using PPBuilder (joined peptide segments).
    Non-AA residues are ignored.
    """
    ppb = PPBuilder()
    seqs: Dict[str, str] = {}
    model = next(iter(struct))  # first model only
    for chain in model:
        polypeps = list(ppb.build_peptides(chain, aa_only=True))
        if not polypeps:
            continue
        seq = "".join(str(pp.get_sequence()) for pp in polypeps)
        if seq:
            seqs[chain.id] = seq
    return seqs

def extract_ca_coords_by_chain(struct) -> Dict[str, List[Tuple[int, np.ndarray]]]:
    """Return per-chain list of (seq_index, CA_xyz) matched to PPBuilder order."""
    ppb = PPBuilder()
    chain_coords = {}
    model = next(iter(struct))
    for chain in model:
        polypeps = list(ppb.build_peptides(chain, aa_only=True))
        if not polypeps:
            continue
        coords = []
        offset = 0
        for pp in polypeps:
            for i, res in enumerate(pp):
                try:
                    ca = res["CA"]
                except KeyError:
                    ca = None
                if ca is not None:
                    coords.append((offset + i, ca.coord.copy()))
            offset += len(pp)
        if coords:
            chain_coords[chain.id] = coords
    return chain_coords

def contact_matrix_from_coords(L: int, coords: List[Tuple[int, np.ndarray]], thresh: float = 8.0):
    """Compute LxL boolean contact map and valid mask from sparse CA coords."""
    has = np.zeros(L, dtype=bool)
    xyz = np.zeros((L, 3), dtype=float)
    for i, c in coords:
        has[i] = True
        xyz[i] = c

    idx = np.where(has)[0]
    contact = np.zeros((L, L), dtype=bool)
    valid = np.zeros((L, L), dtype=bool)

    if len(idx) > 0:
        sub = xyz[idx]
        d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
        c = d < thresh
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                contact[ia, ib] = c[a, b]
                valid[ia, ib] = True
    return contact, valid
