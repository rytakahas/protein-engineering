import torch
from rescontact.models.contact_net import BilinearContactNet

def test_forward_smoke():
    model = BilinearContactNet(embed_dim=320, hidden_dim=64, distance_bias_max=64)
    x = torch.randn(50, 320)
    y = model(x)
    assert y.shape == (50, 50)

