import numpy as np

def _bin_edges_quantile(x, n_bins=10):
    # quantile bins from baseline to ensure non-empty bins
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:
        # fallback to min/median/max
        edges = np.unique([x.min(), np.median(x), x.max()])
    return edges

def psi_from_arrays(baseline, current, n_bins=10, method="quantile", eps=1e-8):
    baseline = np.asarray(baseline).astype(float)
    current  = np.asarray(current).astype(float)
    baseline = baseline[np.isfinite(baseline)]
    current  = current[np.isfinite(current)]
    if baseline.size == 0 or current.size == 0:
        return np.nan

    if method == "quantile":
        edges = _bin_edges_quantile(baseline, n_bins)
    else:
        lo, hi = np.nanmin(baseline), np.nanmax(baseline)
        edges = np.linspace(lo, hi, n_bins+1)

    # clip to ensure within edges
    be = np.histogram(np.clip(baseline, edges[0], edges[-1]), bins=edges)[0].astype(float)
    ce = np.histogram(np.clip(current,  edges[0], edges[-1]), bins=edges)[0].astype(float)
    be = be / max(be.sum(), eps)
    ce = ce / max(ce.sum(), eps)

    # avoid zeroes
    be = np.clip(be, eps, 1)
    ce = np.clip(ce, eps, 1)
    return float(np.sum((ce - be) * np.log(ce / be)))

