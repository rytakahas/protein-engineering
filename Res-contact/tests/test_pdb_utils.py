import numpy as np
from rescontact.data.pdb_utils import contact_map_from_coords

def test_contact_map_shape():
    coords = np.random.randn(10, 3).astype(np.float32)
    y, m = contact_map_from_coords(coords, threshold=8.0)
    assert y.shape == (10, 10)
    assert m.shape == (10, 10)

