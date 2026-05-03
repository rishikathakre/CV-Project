"""
Per-class classification metrics for the four alert levels.

classification_report(y_true, y_pred) returns per-class precision/recall/F1
and macro-averaged totals — the same format as sklearn.metrics.classification_report
but without the sklearn dependency, and restricted to the four alert levels
(NONE / LOW / MEDIUM / HIGH) used by this project.
"""

from __future__ import annotations

from typing import Sequence

ALERT_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, dict]:
    """
    Compute per-class and macro-averaged precision, recall, and F1.

    Args:
        y_true: Ground-truth alert levels per person.
        y_pred: Predicted alert levels per person.

    Returns:
        Dict keyed by class name (and "macro avg").  Each value is:
            {"precision": float, "recall": float, "f1": float, "support": int}
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    classes = sorted(
        set(y_true) | set(y_pred),
        key=lambda c: ALERT_LEVELS.index(c) if c in ALERT_LEVELS else 99,
    )

    report: dict[str, dict] = {}
    for cls in classes:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        support = sum(t == cls for t in y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        report[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        }

    # Macro average — unweighted mean over classes that have at least one
    # ground-truth sample (avoids inflating F1 with zero-support classes).
    active = [v for v in report.values() if v["support"] > 0]
    if active:
        report["macro avg"] = {
            "precision": round(sum(v["precision"] for v in active) / len(active), 4),
            "recall":    round(sum(v["recall"]    for v in active) / len(active), 4),
            "f1":        round(sum(v["f1"]        for v in active) / len(active), 4),
            "support":   len(y_true),
        }

    return report
