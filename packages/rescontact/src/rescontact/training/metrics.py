import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


class ContactMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.probs = []   # list of [L,L] float arrays
        self.targets = [] # list of [L,L] float arrays in {0,1}
        self.masks = []   # list of [L,L] float arrays in {0,1}

    def add(self, probs: np.ndarray, targets: np.ndarray, mask: np.ndarray):
        """Store full matrices; we'll slice/mask as needed later."""
        self.probs.append(probs)
        self.targets.append(targets)
        self.masks.append(mask)

    # -------------------------
    # helpers
    # -------------------------
    def _stack(self):
        """Flatten across all examples using mask (no i<j restriction)."""
        P = np.concatenate([p[m.astype(bool)] for p, m in zip(self.probs, self.masks)]) if self.probs else np.array([])
        T = np.concatenate([t[m.astype(bool)] for t, m in zip(self.targets, self.masks)]) if self.targets else np.array([])
        return P, T

    def _iter_upper(self):
        """Yield (p_upper, t_upper) flattened over i<j & mask for each example."""
        for p, t, m in zip(self.probs, self.targets, self.masks):
            L = p.shape[0]
            iu = np.triu_indices(L, k=1)
            mm = m[iu].astype(bool)
            if mm.sum() == 0:
                yield np.array([]), np.array([])
            else:
                yield p[iu][mm], t[iu][mm]

    def _iter_upper_with_sep(self, min_sep=6):
        """Yield (p_sel, t_sel) for i<j, |i-j|>=min_sep & mask."""
        for p, t, m in zip(self.probs, self.targets, self.masks):
            L = p.shape[0]
            i, j = np.triu_indices(L, k=1)
            sep = (j - i)  # since i<j
            sel = (sep >= int(min_sep))
            if not sel.any():
                yield np.array([]), np.array([])
                continue
            mm = m[i, j].astype(bool)
            both = sel & mm
            if not both.any():
                yield np.array([]), np.array([])
                continue
            yield p[i, j][both], t[i, j][both]

    # -------------------------
    # core metrics
    # -------------------------
    def roc_auc(self):
        P, T = self._stack()
        try:
            return float(roc_auc_score(T, P))
        except Exception:
            return float("nan")

    def pr_auc(self):
        """Average Precision (PR-AUC) over all masked pairs."""
        P, T = self._stack()
        try:
            return float(average_precision_score(T, P))
        except Exception:
            return float("nan")

    def f1(self, threshold: float = 0.5):
        P, T = self._stack()
        if P.size == 0:
            return float("nan")
        pred = (P >= threshold).astype(int)
        try:
            return float(f1_score(T.astype(int), pred))
        except Exception:
            return float("nan")

    # -------------------------
    # precision@k family
    # -------------------------
    def p_at_l(self):
        """Precision@L on upper triangle per example, then average."""
        vals = []
        for p_u, t_u in self._iter_upper():
            if p_u.size == 0:
                continue
            L_est = int(round(np.sqrt(p_u.size * 2)))  # rough; but we want k≈L
            k = max(1, L_est)
            k = min(k, p_u.size)
            idx = np.argsort(-p_u)[:k]
            vals.append(float(t_u[idx].sum()) / float(len(idx)))
        return float(np.mean(vals)) if vals else float("nan")

    def p_at_frac(self, frac=1.0):
        """Precision@floor(frac*L) on upper triangle per example, then average."""
        vals = []
        for p_u, t_u in self._iter_upper():
            if p_u.size == 0:
                continue
            # Reconstruct L from p_u length (n = L*(L-1)/2) approximately
            # Solve L^2 - L - 2n ≈ 0 -> L ≈ (1+sqrt(1+8n))/2
            n = p_u.size
            L_est = int((1.0 + np.sqrt(1.0 + 8.0 * n)) / 2.0)
            k = max(1, int(np.floor(frac * L_est)))
            k = min(k, n)
            idx = np.argsort(-p_u)[:k]
            vals.append(float(t_u[idx].sum()) / float(len(idx)))
        return float(np.mean(vals)) if vals else float("nan")

    def p_at_l2(self):
        return self.p_at_frac(0.5)

    def p_at_l5(self):
        return self.p_at_frac(0.2)

