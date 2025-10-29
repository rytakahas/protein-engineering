from __future__ import annotations
import torch
import torch.nn as nn


class BilinearContactNet(nn.Module):
    """
    Memory-lean model:
      logits[i,j] = h_i^T W h_j + b[|i-j|]
    where h = MLP(emb).
    """
    def __init__(self, embed_dim: int, hidden_dim: int, distance_bias_max: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bilinear = nn.Linear(hidden_dim, hidden_dim, bias=False)  # acts as W
        self.dist_bias = nn.Embedding(distance_bias_max, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: [L, D]
        h = self.proj(emb)         # [L, H]
        Wh = self.bilinear(h)      # [L, H]
        logits = torch.matmul(Wh, h.t())  # [L, L]
        L = h.size(0)
        i = torch.arange(L, device=h.device)
        d = torch.abs(i[:, None] - i[None, :]).clamp(max=self.dist_bias.num_embeddings - 1)
        logits = logits + self.dist_bias(d).squeeze(-1)
        logits.fill_diagonal_(0.0)
        return logits

