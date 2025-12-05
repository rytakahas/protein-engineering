# src/rescontact/utils/psi.py
import numpy as np
from typing import Tuple, Dict

EPS = 1e-12

def quantile_bins(x: np.ndarray, k: int = 10) -> np.ndarray:
    """Return (k) quantile-based bin edges (length k+1)."""
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0, 1.0])
    qs = np.linspace(0, 1, k + 1)
    edges = np.quantile(x, qs)
    # make edges strictly increasing (guard for ties)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([edges[0] - 1e-6, edges[0] + 1e-6])
    return edges

def hist_proportions(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram proportions in given edges."""
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(len(edges) - 1, 1.0 / (len(edges) - 1))
    h, _ = np.histogram(x, bins=edges)
    p = h.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return np.full_like(p, 1.0 / len(p), dtype=np.float64)
    return p / s

def psi_from_props(p: np.ndarray, q: np.ndarray) -> float:
    """
    Population Stability Index between baseline p and current q.
    PSI = sum((p - q) * ln(p/q))
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, EPS, 1.0)  # avoid div-by-zero & log(0)
    q = np.clip(q, EPS, 1.0)
    return float(np.sum((p - q) * np.log(p / q)))

def psi(x_base: np.ndarray, x_cur: np.ndarray, edges: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    p = hist_proportions(x_base, edges)
    q = hist_proportions(x_cur, edges)
    return psi_from_props(p, q), p, q

def categorize_psi(val: float) -> str:
    # common rules of thumb
    if val < 0.1:   return "stable"
    if val < 0.25:  return "slight_shift"
    if val < 0.5:   return "moderate_shift"
    return "major_shift"

