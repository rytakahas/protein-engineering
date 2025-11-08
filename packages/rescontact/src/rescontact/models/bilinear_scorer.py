from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

class BilinearScorer(nn.Module):
    """
    Low-rank bilinear scorer for pairwise contacts, trained on *pairs* (O(P)).
    logits(i,j) = <U h_i, V h_j> + b
    """
    def __init__(self, d_in: int, rank: int = 128):
        super().__init__()
        self.U = nn.Linear(d_in, rank, bias=False)
        self.V = nn.Linear(d_in, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward_pairs(self, H: torch.Tensor, idx_np: np.ndarray) -> torch.Tensor:
        i = torch.as_tensor(idx_np[:,0], device=H.device, dtype=torch.long)
        j = torch.as_tensor(idx_np[:,1], device=H.device, dtype=torch.long)
        A = self.U(H); B = self.V(H)
        return (A[i] * B[j]).sum(-1) + self.bias

    def forward_pairs_two(self, HA: torch.Tensor, HB: torch.Tensor, pairs_AB: np.ndarray) -> torch.Tensor:
        device = HA.device
        i = torch.as_tensor(pairs_AB[:,0], device=device, dtype=torch.long)
        j = torch.as_tensor(pairs_AB[:,1], device=device, dtype=torch.long)
        UA = self.U(HA)
        VB = self.V(HB)
        if VB.device != device:
            VB = VB.to(device)
        return (UA[i] * VB[j]).sum(-1) + self.bias

