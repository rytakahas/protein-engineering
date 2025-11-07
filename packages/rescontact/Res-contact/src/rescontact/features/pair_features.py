
import torch

def build_pair_features(H: torch.Tensor) -> torch.Tensor:
    """Return (L,L,4d) pairwise features: [hi, hj, |hi-hj|, hi*hj]."""
    L, d = H.shape
    hi = H[:, None, :].expand(L, L, d)
    hj = H[None, :, :].expand(L, L, d)
    feat = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj], dim=-1)
    return feat
