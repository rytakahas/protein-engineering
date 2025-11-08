
import torch
import torch.nn as nn

class ContactHead(nn.Module):
    """Linear→ReLU→Dropout→Bilinear + learnable distance bias over |i−j| bins.

    Args:
        input_dim:  feature dimension of per-residue embeddings (e.g., 320 or 341).
        hidden_dim: hidden size before projection.
        rank:       low-rank projection size for bilinear scorer (A @ B^T with A,B in R^{L×rank}).
        dropout:    dropout probability applied after ReLU.
        dist_bins:  number of |i−j| bins for distance bias.
        symmetric:  if True, average A@B^T with (A@B^T)^T to enforce symmetry.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, rank: int = 128,
                 dropout: float = 0.1, dist_bins: int = 512, symmetric: bool = True):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Low-rank projections
        self.proj_u = nn.Linear(hidden_dim, rank, bias=False)
        self.proj_v = nn.Linear(hidden_dim, rank, bias=False)
        # Distance-bias table over |i-j| (clipped to dist_bins-1)
        self.dist_bias = nn.Parameter(torch.zeros(dist_bins))
        nn.init.zeros_(self.dist_bias)
        self.dist_bins = dist_bins
        self.symmetric = symmetric

    def forward(self, H):
        """Compute pairwise logits for an L×d feature matrix H.

        Returns:
            logits: (L, L) pairwise scores before sigmoid.
        """
        L, d = H.shape
        Z = self.pre(H)                 # (L, hidden_dim)
        A = self.proj_u(Z)              # (L, r)
        B = self.proj_v(Z)              # (L, r)

        logits = A @ B.T                # (L, L)
        if self.symmetric:
            logits = 0.5 * (logits + logits.T)

        # Add distance bias by |i-j|
        device = H.device
        idx_i = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
        idx_j = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
        sep = (idx_i - idx_j).abs().clamp_max(self.dist_bins - 1)
        logits = logits + self.dist_bias[sep]
        return logits
