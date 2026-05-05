from __future__ import annotations

from typing import Sequence

# The four possible alert levels in severity order.
ALERT_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]


def classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict]:
    """Compute precision, recall, F1, and support for each alert level.

    Returns a dict keyed by class name plus a 'macro avg' entry.
    The macro average is computed only over classes that appear in y_true.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    # Sort classes by severity so the report is always in the same order.
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

    # Macro average only counts classes that actually appear in y_true.
    active = [v for v in report.values() if v["support"] > 0]
    if active:
        report["macro avg"] = {
            "precision": round(sum(v["precision"] for v in active) / len(active), 4),
            "recall":    round(sum(v["recall"]    for v in active) / len(active), 4),
            "f1":        round(sum(v["f1"]        for v in active) / len(active), 4),
            "support":   len(y_true),
        }

    return report
