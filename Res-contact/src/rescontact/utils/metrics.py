
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def evaluate_pairs(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(int)
    try:
        ap = average_precision_score(y, probs)
        roc = roc_auc_score(y, probs)
    except Exception:
        ap, roc = float("nan"), float("nan")
    return {"pr_auc": ap, "roc_auc": roc}

def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)
    k = max(1, min(len(order), k))
    sel = order[:k]
    return float(labels[sel].mean()) if len(sel) else float("nan")
