import numpy as np
from rescontact.features.pair_features import distance_buckets

def test_distance_buckets():
    d = distance_buckets(5, max_bucket=4)
    assert d.shape == (5, 5)
    assert d.max() <= 3

