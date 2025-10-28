
from typing import Optional, List, Dict
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def sample_pairs(valid_mask: np.ndarray, max_pairs: Optional[int]) -> np.ndarray:
    idx = np.argwhere(valid_mask)
    if idx.size == 0:
        return idx
    if (max_pairs is not None) and (len(idx) > max_pairs):
        sel = np.random.choice(len(idx), size=max_pairs, replace=False)
        idx = idx[sel]
    return idx

def build_pair_batch(H: torch.Tensor, pair_idx: np.ndarray) -> torch.Tensor:
    if pair_idx.size == 0:
        return torch.empty(0, device=DEVICE)
    hi = H[pair_idx[:, 0]]
    hj = H[pair_idx[:, 1]]
    feats = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)
    return feats
