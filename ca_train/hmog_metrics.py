from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve


def _eer_and_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Return (eer, eer_threshold) where threshold is on the `scores` scale."""
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
    except ValueError:
        return 0.0, 0.0

    if len(thresholds) == 0:
        return 0.0, 0.0

    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        eer_threshold = float(interp1d(fpr, thresholds)(eer))
    except Exception:
        # Fallback: choose the threshold where |FPR-FNR| is minimal.
        fnr = 1.0 - tpr
        idx = int(np.argmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        eer_threshold = float(thresholds[idx])

    if np.isnan(eer_threshold):
        eer_threshold = float(thresholds[0])
    return float(eer), float(eer_threshold)


def _confusion_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(np.int32)
    neg = labels == 0
    pos = labels == 1

    tp = int(((preds == 1) & pos).sum())
    fp = int(((preds == 1) & neg).sum())
    tn = int(((preds == 0) & neg).sum())
    fn = int(((preds == 0) & pos).sum())
    pos_n = int(pos.sum())
    neg_n = int(neg.sum())

    far = float(fp / max(neg_n, 1))
    frr = float(fn / max(pos_n, 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))

    # "Impostor detection" view (predict impostor when score < threshold).
    imp_tp = tn  # correctly reject impostor
    imp_fp = fn  # reject genuine
    imp_fn = fp  # accept impostor
    imp_precision = float(imp_tp / max(imp_tp + imp_fp, 1))
    imp_recall = float(imp_tp / max(imp_tp + imp_fn, 1))
    imp_f1 = float(2 * imp_precision * imp_recall / max(imp_precision + imp_recall, 1e-12))

    return {
        "pos": float(pos_n),
        "neg": float(neg_n),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "frr": frr,
        "impostor_precision": imp_precision,
        "impostor_recall": imp_recall,
        "impostor_f1": imp_f1,
    }


def _best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Return (threshold, best_f1) that maximizes F1 on the given set.

    Uses sklearn's precision_recall_curve. Classification rule is: score >= threshold -> positive.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.0, 0.0

    pr = precision[:-1]
    rc = recall[:-1]
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def _threshold_at_target_far(labels: np.ndarray, scores: np.ndarray, target_far: float) -> float:
    """
    Choose a threshold that aims to satisfy FAR <= target_far and maximizes TPR under that constraint.
    """
    target_far = float(target_far)
    if target_far < 0.0:
        target_far = 0.0
    if target_far > 1.0:
        target_far = 1.0

    fpr, tpr, thresholds = roc_curve(labels, scores)
    candidates = np.where(fpr <= target_far)[0]
    if candidates.size == 0:
        # No achievable threshold; fall back to the strictest threshold.
        return float(thresholds[0])

    best = int(candidates[np.argmax(tpr[candidates])])
    return float(thresholds[best])


def _threshold_at_target_frr(labels: np.ndarray, scores: np.ndarray, target_frr: float) -> float:
    """
    Choose a threshold that aims to satisfy FRR <= target_frr and minimizes FAR under that constraint.

    FRR = 1 - TPR. We search ROC curve points and pick the candidate with:
      - fnr <= target_frr
      - minimal fpr; if ties, pick the strictest (highest) threshold.
    """
    target_frr = float(target_frr)
    if target_frr < 0.0:
        target_frr = 0.0
    if target_frr > 1.0:
        target_frr = 1.0

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    candidates = np.where(fnr <= target_frr)[0]
    if candidates.size == 0:
        # Cannot reach target; pick the point with minimal FRR (max TPR).
        best = int(np.argmax(tpr))
        return float(thresholds[best])

    # Among candidates, prefer minimal FAR; among ties, prefer higher threshold.
    candidate_fpr = fpr[candidates]
    min_fpr = float(candidate_fpr.min())
    min_candidates = candidates[np.where(candidate_fpr == min_fpr)[0]]
    best = int(min_candidates[np.argmax(thresholds[min_candidates])])
    return float(thresholds[best])


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: Optional[float] = None,
    threshold_strategy: str = "eer",
    target_far: Optional[float] = None,
    target_frr: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute authentication metrics from binary labels and real-valued scores.

    Conventions:
    - labels: 1 = genuine, 0 = impostor
    - scores: larger means "more genuine"
    """
    labels = np.asarray(labels).astype(np.int32, copy=False)
    scores = np.asarray(scores).astype(np.float64, copy=False)

    if labels.size == 0:
        return {
            "auc": 0.0,
            "far": 0.0,
            "frr": 0.0,
            "eer": 0.0,
            "f1": 0.0,
            "threshold": 0.0,
            "eer_threshold": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "impostor_precision": 0.0,
            "impostor_recall": 0.0,
            "impostor_f1": 0.0,
        }

    if np.unique(labels).size < 2:
        # Undefined for ROC-based metrics; return safe defaults to avoid crashing.
        base = {
            "auc": 0.0,
            "far": 0.0,
            "frr": 0.0,
            "eer": 0.0,
            "f1": 0.0,
            "threshold": 0.0 if threshold is None else float(threshold),
            "eer_threshold": 0.0,
        }
        conf = _confusion_at_threshold(labels, scores, float(base["threshold"]))
        return {**base, **conf}

    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = 0.0

    eer, eer_threshold = _eer_and_threshold(labels, scores)

    selected_threshold: float
    best_f1_threshold: Optional[float] = None
    best_f1: Optional[float] = None
    if threshold is not None:
        selected_threshold = float(threshold)
    else:
        strategy = str(threshold_strategy or "eer").lower().strip()
        if strategy == "eer":
            selected_threshold = float(eer_threshold)
        elif strategy == "f1":
            best_f1_threshold, best_f1 = _best_f1_threshold(labels, scores)
            selected_threshold = float(best_f1_threshold)
        elif strategy == "far":
            if target_far is None:
                raise ValueError("threshold_strategy='far' requires target_far")
            selected_threshold = float(_threshold_at_target_far(labels, scores, float(target_far)))
        elif strategy == "frr":
            if target_frr is None:
                raise ValueError("threshold_strategy='frr' requires target_frr")
            selected_threshold = float(_threshold_at_target_frr(labels, scores, float(target_frr)))
        else:
            raise ValueError(f"Unknown threshold_strategy={threshold_strategy!r}")

    conf = _confusion_at_threshold(labels, scores, selected_threshold)
    metrics = {**conf}
    metrics.update(
        {
            "auc": float(auc),
            "eer": float(eer),
            "threshold": selected_threshold,
            "eer_threshold": float(eer_threshold),
        }
    )
    if best_f1_threshold is not None:
        metrics["best_f1_threshold"] = float(best_f1_threshold)
    if best_f1 is not None:
        metrics["best_f1"] = float(best_f1)
    return metrics
