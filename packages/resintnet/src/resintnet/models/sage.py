# src/resintnet/models/sage.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import List, Tuple

Tensor = torch.Tensor
EdgeList = List[Tuple[int, int, int]]

class GraphBatchNetAMP(nn.Module):
    """
    Lightweight message-passing regressor with an explicit g_mem gate:
      - Edge messages are multiplied by a positive gate derived ONLY from g_mem (E[:, 2]).
      - Readout concatenates node context (mean over nodes) + edge context (gated mean over edges).
      - Designed to keep ∂Â/∂g_mem ≠ 0, avoid saturation (softplus), and run on CPU.
    Outputs: [A_hat, ddG_hat] per graph.
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64, edge_dropout: float = 0.1):
        super().__init__()
        H = hidden
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
        )
        # uses (xi, xj, e_ij) as input
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
        )
        # add direct edge context path
        self.edge_proj = nn.Linear(edge_dim, H)

        # readout over [node_ctx || edge_ctx]
        self.read = nn.Sequential(
            nn.Linear(H + H, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        # learnable gate parameters (non-saturating via softplus)
        self.gate_scale = nn.Parameter(torch.tensor(8.0))    # steeper -> more sensitive to g_mem
        self.g_gate_center = nn.Parameter(torch.tensor(0.35)) # center in [0,1] g_mem range

        self.edge_dropout = float(edge_dropout)

    def forward(self, batch_feats: List[Tuple[Tensor, EdgeList, Tensor]]) -> Tensor:
        """
        batch_feats: list of (X, edges, E) for each graph in batch
          X: [N, Fx]  E: [M, Fe]  edges: list of tuples (i,j,etype)
        returns: [B, 2]
        """
        outs = []
        for (X, edges, E) in batch_feats:
            outs.append(self.forward_one(X, edges, E))
        return torch.stack(outs, dim=0) if len(outs) else torch.zeros((0, 2), dtype=torch.float32)

    def forward_one(self, X: Tensor, edges: EdgeList, E: Tensor) -> Tensor:
        """
        One graph forward.
        """
        device = X.device
        Hx = self.node_mlp(X)
        Hx = torch.relu(Hx)

        # g_mem gate (column 2). If absent, fall back to zeros -> small gate via softplus + 1e-3
        if E.shape[1] >= 3:
            g_raw = E[:, 2:3]
        else:
            g_raw = torch.zeros((E.shape[0], 1), device=device, dtype=E.dtype)

        # non-saturating, strictly-positive gate
        gate = torch.nn.functional.softplus(self.gate_scale * (g_raw - self.g_gate_center)) + 1e-3  # [M,1]

        agg = torch.zeros_like(Hx)

        # optional edge dropout to prevent trivial shortcuts
        if self.training and self.edge_dropout > 0.0 and len(edges) > 0:
            keep_mask = (torch.rand((len(edges),), device=device) > self.edge_dropout)
        else:
            keep_mask = None

        for k, (i, j, _) in enumerate(edges):
            if keep_mask is not None and not bool(keep_mask[k]):
                continue
            # edge embedding uses both nodes and full edge feature vector
            e_in = torch.cat([X[i], X[j], E[k]], dim=0)
            e_ij = self.edge_mlp(e_in)
            e_ij = torch.relu(e_ij)
            # apply the g_mem gate (scalar per edge)
            m_ij = gate[k] * e_ij  # [H]
            # symmetric aggregation
            agg[i] = agg[i] + m_ij
            agg[j] = agg[j] + m_ij

        Hn = Hx + agg

        # edge context (weighted by gate), averaged
        if len(edges) > 0:
            gated_proj_sum = torch.zeros_like(Hx[0])
            for k in range(len(edges)):
                gated_proj_sum = gated_proj_sum + gate[k] * self.edge_proj(E[k])
            edge_ctx = gated_proj_sum / (len(edges) + 1e-6)
        else:
            edge_ctx = torch.zeros_like(Hx[0])

        node_ctx = Hn.mean(dim=0)  # [H]
        out = self.read(torch.cat([node_ctx, edge_ctx], dim=0))  # [2]
        return out

