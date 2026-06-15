import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix,
)

def core_metrics(y_true, y_score) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc":  float(average_precision_score(y_true, y_score)),  # = area under PR curve
        "prevalence": float(np.mean(y_true)),
    }

def precision_at_k(y_true, y_score, k_values) -> dict:
    """Of the k highest-scored transactions, what fraction are fraud?
    Answers: 'if ops reviews k cases/day, what's the hit rate?'"""
    order = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true)[order]
    out = {}
    for k in k_values:
        k = min(k, len(y_sorted))
        flagged = y_sorted[:k]
        out[f"precision_at_{k}"] = float(flagged.mean())
        out[f"recall_at_{k}"] = float(flagged.sum() / y_sorted.sum())
    return out

def cost_optimal_threshold(y_true, y_score, cost_fn, cost_fp) -> dict:
    """Pick the threshold minimizing expected cost = cost_fn*FN + cost_fp*FP.
    The honest alternative to a default 0.5 cutoff."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    y_true = np.asarray(y_true)
    total_pos = y_true.sum()
    best = {"threshold": 0.5, "cost": np.inf}
    # precision_recall_curve drops the last threshold; align by iterating thr.
    for t in np.unique(thr):
        pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = cost_fn * fn + cost_fp * fp
        if cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost),
                    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                    "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                    "recall": float(tp / total_pos) if total_pos else 0.0}
    return best