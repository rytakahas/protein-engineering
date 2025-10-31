# tests/test_pair_features.py
import torch
import numpy as np
from rescontact.features.pair_features import build_pair_features

def test_build_pair_features_shape_and_content():
    L, d = 5, 8
    H = torch.randn(L, d)
    F = build_pair_features(H)  # [L, L, 4d]

    # shape & dtype
    assert F.shape == (L, L, 4 * d)
    assert isinstance(F, torch.Tensor)

    # Slices
    hi   = F[:, :, :d]         # should repeat H[i] along j
    hj   = F[:, :, d:2*d]      # should repeat H[j] along i
    diff = F[:, :, 2*d:3*d]    # |H[i]-H[j]|
    prod = F[:, :, 3*d:4*d]    # H[i]*H[j]

    # hi equals H broadcast over j
    for i in range(L):
        assert torch.allclose(hi[i], H[i].expand(L, d))

    # hj equals H broadcast over i
    for j in range(L):
        assert torch.allclose(hj[:, j], H[j].expand(L, d))

    # symmetry checks for |hi-hj| and hi*hj|
    for i in range(L):
        for j in range(L):
            assert torch.allclose(diff[i, j], diff[j, i])  # |a-b| == |b-a|
            assert torch.allclose(prod[i, j], prod[j, i])  # a*b == b*a

    # diagonal properties
    for i in range(L):
        # |H[i]-H[i]| == 0
        assert torch.allclose(diff[i, i], torch.zeros(d))
        # H[i]*H[i] == H[i]^2 (non-neg per element)
        assert torch.all(prod[i, i] >= 0)

def test_build_pair_features_grad_flow():
    """Light autograd smoke test to ensure features are connected back to H."""
    L, d = 4, 6
    H = torch.randn(L, d, requires_grad=True)
    F = build_pair_features(H)
    # Use a scalar depending on all parts to check gradients flow
    loss = F.sum()
    loss.backward()
    assert H.grad is not None
    # grad should be finite and of the same shape
    assert H.grad.shape == H.shape
    assert torch.isfinite(H.grad).all()

