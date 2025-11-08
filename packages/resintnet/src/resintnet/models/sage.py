
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


def _to_torch_sparse(adj: sp.csr_matrix) -> torch.Tensor:
    adj = adj.tocoo()
    indices = torch.tensor([adj.row, adj.col], dtype=torch.long)
    values  = torch.tensor(adj.data, dtype=torch.float32)
    shape   = torch.Size(adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def normalize_adj(adj: sp.csr_matrix, self_loops: bool = True) -> sp.csr_matrix:
    if self_loops:
        adj = adj + sp.eye(adj.shape[0], format="csr")
    deg = (adj.sum(1).A.ravel()).astype("float32")
    deg_inv_sqrt = (deg + 1e-8) ** -0.5
    D_inv = sp.diags(deg_inv_sqrt)
    return D_inv @ adj @ D_inv


class SimpleSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        h1 = torch.sparse.mm(A_hat, x)
        h1 = F.relu(self.w1(h1))
        h2 = torch.sparse.mm(A_hat, h1)
        h2 = F.relu(self.w2(h2))
        z  = self.out(h2).squeeze(-1)
        return z


def score_nodes(x: torch.Tensor, adj_norm: sp.csr_matrix, hidden: int = 128, device: str = "cpu") -> torch.Tensor:
    A_hat = _to_torch_sparse(adj_norm).coalesce().to(device)
    model = SimpleSAGE(x.shape[1], hidden=hidden).to(device)
    with torch.no_grad():
        z = model(x.to(device), A_hat)
    return z.cpu()
