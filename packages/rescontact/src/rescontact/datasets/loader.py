
from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from Bio.PDB import MMCIFParser, PDBParser, PPBuilder
import os

def _is_cif(path: str) -> bool:
    p = path.lower()
    return p.endswith(".cif") or p.endswith(".cif.gz") or p.endswith(".bcif")

def _open_handle(path: str):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt")
    return open(path, "rt")

def _load_structure(path: str):
    if _is_cif(path):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    # Parsers read from filename directly
    return parser.get_structure(os.path.basename(path), path)

def load_chains(path: str, max_len_per_chain: Optional[int] = None) -> Dict[str, Dict]:
    st = _load_structure(path)
    ppb = PPBuilder()
    out: Dict[str, Dict] = {}
    for model in st:
        for chain in model:
            chain_id = chain.id
            residues = [res for res in chain if res.id[0] == ' ']
            # Sequence via polypeptide builder
            seq = ""
            for pp in ppb.build_peptides(chain):
                seq += str(pp.get_sequence())
            if not seq:
                continue
            ca_list, mask = [], []
            for res in residues:
                if 'CA' in res:
                    ca_list.append(res['CA'].coord.astype(np.float32))
                    mask.append(1)
                else:
                    ca_list.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                    mask.append(0)
            L = min(len(ca_list), len(seq))
            ca_xyz = np.array(ca_list[:L], dtype=np.float32)
            mask = np.array(mask[:L], dtype=np.uint8)
            if max_len_per_chain is not None:
                ca_xyz = ca_xyz[:max_len_per_chain]
                mask = mask[:max_len_per_chain]
                seq = seq[:max_len_per_chain]
            out[chain_id] = dict(seq=seq, ca_xyz=ca_xyz, mask=mask)
        break
    return out

def concat_chains(chains: Dict[str, Dict], include_inter_chain: bool = True) -> Dict[str, np.ndarray]:
    seqs, coords, masks, chain_idx = [], [], [], []
    for k, (cid, d) in enumerate(chains.items()):
        s = d['seq']; x = d['ca_xyz']; m = d['mask']
        seqs.append(s); coords.append(x); masks.append(m)
        chain_idx.append(np.full(len(s), k, dtype=np.int32))
    seq = "".join(seqs)
    ca_xyz = np.concatenate(coords, axis=0) if coords else np.zeros((0,3), np.float32)
    mask = np.concatenate(masks, axis=0) if masks else np.zeros((0,), np.uint8)
    chain_idx = np.concatenate(chain_idx, axis=0) if chain_idx else np.zeros((0,), np.int32)
    L = len(seq)
    valid = np.ones((L, L), dtype=np.uint8)
    np.fill_diagonal(valid, 0)
    if L:
        miss = (mask == 0).astype(np.uint8)
        valid *= np.outer(1 - miss, 1 - miss)
    if not include_inter_chain and L:
        valid *= (np.equal.outer(chain_idx, chain_idx)).astype(np.uint8)
    return dict(seq=seq, ca_xyz=ca_xyz, mask=mask, chain_idx=chain_idx, valid_pairs=valid)

def contact_labels(ca_xyz: np.ndarray, thresh: float = 8.0) -> np.ndarray:
    L = ca_xyz.shape[0]
    contacts = np.zeros((L, L), dtype=np.uint8)
    ok = ~np.isnan(ca_xyz).any(axis=1)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return contacts
    X = ca_xyz[idx]
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    C = (D <= thresh).astype(np.uint8)
    contacts[np.ix_(idx, idx)] = C
    np.fill_diagonal(contacts, 0)
    return contacts
