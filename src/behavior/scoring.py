import numpy as np

from shared.data_types import BehaviorFeatures

# Weights that control how much each feature contributes to the final score.
# They must add up to 1.0. These defaults come from the grid search results.
ALPHA = 0.30  # dwell time anomaly
BETA  = 0.30  # zone revisits
GAMMA = 0.20  # trajectory irregularity
DELTA = 0.20  # billing bypass

# Default dwell threshold: flag someone who lingers more than 60 seconds in one zone.
_DWELL_ANOMALY_THRESHOLD_S = 60.0

# Number of total revisits across all zones at which the revisit score is 1.0.
_REVISIT_SATURATION        = 5.0


def _dwell_f(features: BehaviorFeatures, threshold: float) -> float:
    """Return a 0-1 score based on how long the person stayed in their longest-visited zone."""
    if not features.dwell_per_zone or threshold <= 0:
        return 0.0
    return min(max(features.dwell_per_zone.values()) / threshold, 1.0)


def _revisit_f(features: BehaviorFeatures, saturation: float) -> float:
    """Return a 0-1 score based on total zone revisits across all zones."""
    if saturation <= 0:
        return 0.0
    return min(sum(features.zone_revisits.values()) / saturation, 1.0)


def _apply_weights(f1, f2, f3, f4) -> float:
    """Combine the four feature scores into one weighted sum, clamped to [0, 1]."""
    return max(0.0, min(1.0, ALPHA * f1 + BETA * f2 + GAMMA * f3 + DELTA * f4))


def compute_score(features: BehaviorFeatures) -> BehaviorFeatures:
    """Compute a suspicion score using the fixed default thresholds and return updated features."""
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


class AdaptiveScorer:
    """Scores people relative to others in the current scene instead of fixed thresholds.

    After seeing at least _MIN_SAMPLES people, the scorer computes a threshold
    as mean + K * std of the observed dwell/revisit values. This means thresholds
    automatically adjust to the behaviour of the current crowd.
    """

    _MIN_SAMPLES = 3   # need at least this many people before switching to adaptive mode
    _K           = 1.5 # how many standard deviations above the mean counts as anomalous

    def __init__(self) -> None:
        self._dwell:   dict[int, float] = {}
        self._revisit: dict[int, float] = {}

    def compute(self, features: BehaviorFeatures):
        """Update the population pool with this person and score them."""
        self._dwell[features.id]   = max(features.dwell_per_zone.values(), default=0.0)
        self._revisit[features.id] = float(sum(features.zone_revisits.values()))
        return self._score(features)

    def compute_final(self, features: BehaviorFeatures):
        """Score using the final population pool without adding to it again."""
        return self._score(features)

    def is_calibrated(self) -> bool:
        """Return True if enough people have been seen to use adaptive thresholds."""
        return len(self._dwell) >= self._MIN_SAMPLES

    def n_samples(self) -> int:
        return len(self._dwell)

    def current_thresholds(self) -> dict:
        """Return the current dwell and revisit thresholds for display in the dashboard."""
        return {
            "dwell_s":  round(self._thresh(list(self._dwell.values()),   _DWELL_ANOMALY_THRESHOLD_S), 1),
            "revisits": round(self._thresh(list(self._revisit.values()), _REVISIT_SATURATION), 1),
            "adaptive": self.is_calibrated(),
            "n":        self.n_samples(),
        }

    def _thresh(self, pool: list, fallback: float) -> float:
        """Compute mean + K*std from the pool, or fall back to the default if pool is too small."""
        if len(pool) < self._MIN_SAMPLES:
            return fallback
        arr = np.array(pool, dtype=float)
        t = float(arr.mean() + self._K * arr.std())
        return max(t, fallback * 0.10)

    def _score(self, features: BehaviorFeatures):
        """Score this person using the current adaptive thresholds and return features + breakdown."""
        dt = self._thresh(list(self._dwell.values()),   _DWELL_ANOMALY_THRESHOLD_S)
        rt = self._thresh(list(self._revisit.values()), _REVISIT_SATURATION)

        f1 = _dwell_f(features, dt)
        f2 = _revisit_f(features, rt)
        f3 = features.trajectory_irregularity
        f4 = 1.0 if features.billing_bypassed else 0.0

        score = _apply_weights(f1, f2, f3, f4)

        # Breakdown shows how many points each feature contributed (out of 100).
        breakdown = {
            "Dwell anomaly":     round(f1 * ALPHA * 100),
            "Zone revisits":     round(f2 * BETA  * 100),
            "Path irregularity": round(f3 * GAMMA * 100),
            "Billing bypass":    round(f4 * DELTA * 100),
            "raw":               [f1, f2, f3, f4],
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
