import numpy as np
import torch
from rescontact.models.bilinear_scorer import BilinearScorer

def test_pairs_forward():
    H = torch.randn(50, 64)
    pairs = np.array([[0,1],[2,3],[10,11]], dtype=np.int32)
    m = BilinearScorer(d_in=64, rank=16)
    y = m.forward_pairs(H, pairs)
    assert y.shape == (3,)

