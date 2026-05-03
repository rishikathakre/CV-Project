"""
Suspicion scoring.

compute_score()  — stateless, fixed-threshold scorer (kept for backward compat / tests).
AdaptiveScorer   — stateful scorer that calibrates normalization thresholds from the
                   observed population instead of using hard-coded constants.

Formula (shared by both):
    S = α·f1 + β·f2 + γ·f3 + δ·f4
    f1 = dwell_anomaly          (longest single-zone dwell, normalised)
    f2 = revisit_score          (total zone re-entries, normalised)
    f3 = trajectory_irregularity
    f4 = billing_bypass         (binary)
"""

import numpy as np

from shared.data_types import BehaviorFeatures

# ---------------------------------------------------------------------------
# Shared weights (α β γ δ)
# ---------------------------------------------------------------------------
ALPHA = 0.30   # dwell anomaly
BETA  = 0.30   # revisit score
GAMMA = 0.20   # trajectory irregularity
DELTA = 0.20   # billing bypass

# Fixed fallback thresholds (used when the population is too small to calibrate).
_DWELL_ANOMALY_THRESHOLD_S = 60.0   # 60 s in one zone before score rises
_REVISIT_SATURATION        = 5.0


# ---------------------------------------------------------------------------
# Shared sub-feature helpers
# ---------------------------------------------------------------------------

def _dwell_f(features: BehaviorFeatures, threshold: float) -> float:
    if not features.dwell_per_zone or threshold <= 0:
        return 0.0
    return min(max(features.dwell_per_zone.values()) / threshold, 1.0)


def _revisit_f(features: BehaviorFeatures, saturation: float) -> float:
    if saturation <= 0:
        return 0.0
    return min(sum(features.zone_revisits.values()) / saturation, 1.0)


def _apply_weights(f1, f2, f3, f4) -> float:
    return max(0.0, min(1.0, ALPHA * f1 + BETA * f2 + GAMMA * f3 + DELTA * f4))


# ---------------------------------------------------------------------------
# Stateless scorer — fixed thresholds (kept for backward compat / unit tests)
# ---------------------------------------------------------------------------

def compute_score(features: BehaviorFeatures) -> BehaviorFeatures:
    """
    Compute suspicion score using fixed normalization thresholds.
    Kept for backward compatibility with tests and external callers.
    """
    f1 = _dwell_f(features, _DWELL_ANOMALY_THRESHOLD_S)
    f2 = _revisit_f(features, _REVISIT_SATURATION)
    f3 = features.trajectory_irregularity
    f4 = 1.0 if features.billing_bypassed else 0.0
    score = _apply_weights(f1, f2, f3, f4)

    return BehaviorFeatures(
        id=features.id,
        dwell_per_zone=features.dwell_per_zone,
        zone_revisits=features.zone_revisits,
        zone_sequence=features.zone_sequence,
        billing_bypassed=features.billing_bypassed,
        trajectory_irregularity=features.trajectory_irregularity,
        suspicion_score=score,
        alert_reasons=features.alert_reasons,
    )


# ---------------------------------------------------------------------------
# Adaptive scorer — thresholds calibrate from observed population
# ---------------------------------------------------------------------------

class AdaptiveScorer:
    """
    Maintains a running snapshot of each tracked person's latest behavioral
    values and derives normalization thresholds from the population distribution
    (mean + K·std) rather than fixed constants.

    After _MIN_SAMPLES distinct persons have been observed the scorer switches
    from the fixed fallback thresholds to adaptive ones.  This satisfies the
    project goal: "score weights calibrate to the store's own behavioral
    distribution — no manually defined thresholds."

    Usage:
        scorer = AdaptiveScorer()
        # inside the per-frame loop:
        features, breakdown = scorer.compute(features)
        # after the loop, for final summary:
        features, breakdown = scorer.compute_final(features)
    """

    _MIN_SAMPLES = 3    # minimum distinct persons before adaptive mode engages
    _K           = 1.5  # anomaly multiplier: threshold = mean + K * std

    def __init__(self) -> None:
        # One entry per person ID — overwritten each frame with latest value.
        self._dwell:   dict[int, float] = {}
        self._revisit: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, features: BehaviorFeatures):
        """
        Update population pool from this person's current snapshot, then score.

        Returns (updated_BehaviorFeatures, breakdown) where breakdown is a dict
        mapping feature name → integer contribution to score on 0-100 scale.
        """
        # Update population pool (overwrite, not append — one entry per person).
        self._dwell[features.id]   = max(features.dwell_per_zone.values(), default=0.0)
        self._revisit[features.id] = float(sum(features.zone_revisits.values()))

        return self._score(features)

    def compute_final(self, features: BehaviorFeatures):
        """Score using current thresholds without updating the population pool."""
        return self._score(features)

    def is_calibrated(self) -> bool:
        return len(self._dwell) >= self._MIN_SAMPLES

    def n_samples(self) -> int:
        return len(self._dwell)

    def current_thresholds(self) -> dict:
        """Return the active normalization thresholds (for UI display)."""
        return {
            "dwell_s":  round(self._thresh(list(self._dwell.values()),   _DWELL_ANOMALY_THRESHOLD_S), 1),
            "revisits": round(self._thresh(list(self._revisit.values()), _REVISIT_SATURATION), 1),
            "adaptive": self.is_calibrated(),
            "n":        self.n_samples(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _thresh(self, pool: list, fallback: float) -> float:
        """Compute adaptive threshold; fall back to fixed value if too few samples."""
        if len(pool) < self._MIN_SAMPLES:
            return fallback
        arr = np.array(pool, dtype=float)
        t = float(arr.mean() + self._K * arr.std())
        # Never collapse below 10 % of fallback (avoid divide-by-near-zero).
        return max(t, fallback * 0.10)

    def _score(self, features: BehaviorFeatures):
        dt = self._thresh(list(self._dwell.values()),   _DWELL_ANOMALY_THRESHOLD_S)
        rt = self._thresh(list(self._revisit.values()), _REVISIT_SATURATION)

        f1 = _dwell_f(features, dt)
        f2 = _revisit_f(features, rt)
        f3 = features.trajectory_irregularity
        f4 = 1.0 if features.billing_bypassed else 0.0

        score = _apply_weights(f1, f2, f3, f4)

        breakdown = {
            "Dwell anomaly":     round(f1 * ALPHA * 100),
            "Zone revisits":     round(f2 * BETA  * 100),
            "Path irregularity": round(f3 * GAMMA * 100),
            "Billing bypass":    round(f4 * DELTA * 100),
            "raw":               [f1, f2, f3, f4],   # exact normalized values for SHAP
        }

        updated = BehaviorFeatures(
            id=features.id,
            dwell_per_zone=features.dwell_per_zone,
            zone_revisits=features.zone_revisits,
            zone_sequence=features.zone_sequence,
            billing_bypassed=features.billing_bypassed,
            trajectory_irregularity=features.trajectory_irregularity,
            suspicion_score=score,
            alert_reasons=features.alert_reasons,
        )
        return updated, breakdown
