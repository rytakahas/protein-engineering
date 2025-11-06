# src/rescontact/templates/fuse.py
"""
Fusion utilities to combine ESM logits with template priors.

- logit_blend:  blended = (1 - alpha) * esm_logits + alpha * template_logits
- feature_concat: return concatenated pairwise features; your head must accept them.

In practice, template priors may be provided as either contact logits (L, L)
or as bin logits (L, L, B). This helper keeps it simple.
"""
from typing import Optional
import numpy as np

def logit_blend(esm_logits: np.ndarray,
                template_logits: Optional[np.ndarray],
                alpha: float = 0.3) -> np.ndarray:
    if template_logits is None:
        return esm_logits
    return (1.0 - alpha) * esm_logits + alpha * template_logits

def feature_concat(esm_pair_features: np.ndarray,
                   template_pair_features: Optional[np.ndarray]) -> np.ndarray:
    if template_pair_features is None:
        return esm_pair_features
    return np.concatenate([esm_pair_features, template_pair_features], axis=-1)
