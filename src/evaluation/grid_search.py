from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np

from shared.data_types import BehaviorFeatures
from src.alerts.explainer import get_alert_level
from src.evaluation.metrics import classification_report


def _score_with_weights(raw: Sequence[float], alpha: float, beta: float, gamma: float, delta: float) -> float:
    """Compute the suspicion score using a specific set of weights instead of the defaults."""
    f1, f2, f3, f4 = raw
    return float(np.clip(alpha * f1 + beta * f2 + gamma * f3 + delta * f4, 0.0, 1.0))


def _predict_level(features: BehaviorFeatures, score: float) -> str:
    """Inject a custom score into the features and return the alert level the rules produce."""
    tmp = BehaviorFeatures(
        id=features.id,
        dwell_per_zone=features.dwell_per_zone,
        zone_revisits=features.zone_revisits,
        zone_sequence=features.zone_sequence,
        billing_bypassed=features.billing_bypassed,
        trajectory_irregularity=features.trajectory_irregularity,
        suspicion_score=score,
    )
    return get_alert_level(tmp)


def run_grid_search(
    person_features: dict[int, tuple[BehaviorFeatures, list[float]]],
    ground_truth: dict[int, str],
    step: float = 0.1,
) -> tuple[dict, list[dict]]:
    """Try all weight combinations (alpha, beta, gamma, delta) that sum to 1.0.

    For each combination the alert levels are re-predicted and scored against
    ground truth. Returns the best combination and all results sorted by F1.

    Step size of 0.1 gives 286 combinations.
    """
    pids = [pid for pid in person_features if pid in ground_truth]
    if not pids:
        return {}, []

    y_true = [ground_truth[pid] for pid in pids]
    grid = [round(i * step, 2) for i in range(int(round(1.0 / step)) + 1)]

    results: list[dict] = []
    best_f1 = -1.0
    best_weights: dict = {"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2, "macro_f1": 0.0}

    for a, b, g in product(grid, repeat=3):
        d = round(1.0 - a - b - g, 2)
        if d < 0.0 or d > 1.0:
            continue
        if abs(a + b + g + d - 1.0) > 1e-6:
            continue

        y_pred = [
            _predict_level(person_features[pid][0],
                           _score_with_weights(person_features[pid][1], a, b, g, d))
            for pid in pids
        ]

        report = classification_report(y_true, y_pred)
        macro_f1 = report.get("macro avg", {}).get("f1", 0.0)
        results.append({"alpha": a, "beta": b, "gamma": g, "delta": d, "macro_f1": macro_f1})

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_weights = {"alpha": a, "beta": b, "gamma": g, "delta": d, "macro_f1": macro_f1}

    results.sort(key=lambda x: x["macro_f1"], reverse=True)
    return best_weights, results
