
import torch, torch.nn as nn, torch.nn.functional as F

class GraphBatchNetAMP(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim*2 + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.read = nn.Sequential(
            nn.Linear(hidden*2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.gate_scale = nn.Parameter(torch.tensor(2.0))

    def forward_one(self, X, edges, E):
        Hx = F.relu(self.node_mlp(X))
        g_raw = E[:,2:3] if E.shape[1] >= 3 else torch.zeros((E.shape[0],1), device=E.device, dtype=E.dtype)
        gate = torch.clamp(1.0 + self.gate_scale * g_raw, 0.0, 3.0)

        agg = torch.zeros_like(Hx)
        edge_ctx = torch.zeros_like(Hx[0])
        for k,(i,j,_) in enumerate(edges):
            e_ij = torch.cat([X[i], X[j], E[k]], dim=0)
            m = F.relu(self.edge_mlp(e_ij)) * gate[k]
            agg[i] = agg[i] + m
            agg[j] = agg[j] + m
            edge_ctx = edge_ctx + gate[k] * self.edge_proj(E[k])
        edge_ctx = edge_ctx / (len(edges) + 1e-6)
        H = Hx + agg
        out = self.read(torch.cat([H.mean(dim=0), edge_ctx], dim=0))
        return out

    def forward(self, packed_graphs):
        return torch.stack([self.forward_one(X, edges, E) for (X,edges,E) in packed_graphs])
