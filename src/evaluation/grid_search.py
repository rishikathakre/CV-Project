"""
Grid search over scoring weights (ALPHA, BETA, GAMMA, DELTA) to maximise macro-F1.

The four weights must sum to 1.0 (the constraint enforced here).  With a step
of 0.1 there are ~286 valid combinations — the search completes in well under
a second for a typical clip of 1–5 tracked persons.

Usage:
    from src.evaluation.grid_search import run_grid_search

    # person_features: {pid: (BehaviorFeatures, [f1, f2, f3, f4])}
    # ground_truth:    {pid: "NONE"|"LOW"|"MEDIUM"|"HIGH"}
    best, results = run_grid_search(person_features, ground_truth)
    print(best)   # {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.2, 'delta': 0.1, 'macro_f1': 0.78}
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np

from shared.data_types import BehaviorFeatures
from src.alerts.explainer import get_alert_level
from src.evaluation.metrics import classification_report


def _score_with_weights(
    raw: Sequence[float],
    alpha: float,
    beta:  float,
    gamma: float,
    delta: float,
) -> float:
    f1, f2, f3, f4 = raw
    return float(np.clip(alpha * f1 + beta * f2 + gamma * f3 + delta * f4, 0.0, 1.0))


def _predict_level(features: BehaviorFeatures, score: float) -> str:
    """Derive alert level from rule triggers + the supplied score."""
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
    """
    Exhaustive grid search over (ALPHA, BETA, GAMMA, DELTA) with α+β+γ+δ=1.

    Args:
        person_features: {pid: (BehaviorFeatures, [f1, f2, f3, f4])}
            where [f1..f4] are the final normalized feature values.
        ground_truth: {pid: alert_level_string}
        step: Grid resolution (default 0.1).  Use 0.2 for a faster run.

    Returns:
        (best_weights, all_results_sorted_by_f1)
        best_weights has keys: alpha, beta, gamma, delta, macro_f1
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
