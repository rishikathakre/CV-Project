"""
Adaptive suspicion scoring.

Computes a weighted suspicion score in [0.0, 1.0] from four normalised
sub-features.  Weights are exposed as module-level constants so they can
be tuned without touching the scoring logic.

Formula:
    S = α*f1 + β*f2 + γ*f3 + δ*f4
    f1 = dwell_anomaly       (long dwell in any single zone)
    f2 = revisit_score       (normalised count of zone re-entries)
    f3 = trajectory_irregularity  (from features.py)
    f4 = billing_bypass      (binary: 0.0 or 1.0)
"""

from shared.data_types import BehaviorFeatures

# ---------------------------------------------------------------------------
# Tunable weights (α, β, γ, δ)
# ---------------------------------------------------------------------------
ALPHA = 0.3   # dwell anomaly weight
BETA  = 0.3   # revisit score weight
GAMMA = 0.2   # trajectory irregularity weight
DELTA = 0.2   # billing bypass weight

# Thresholds used for normalisation.
_DWELL_ANOMALY_THRESHOLD_S = 30.0   # seconds in one zone → f1 = 1.0
_REVISIT_SATURATION        = 5      # revisit count that saturates f2 at 1.0


def compute_score(features: BehaviorFeatures) -> BehaviorFeatures:
    """
    Compute suspicion score for the given BehaviorFeatures and return a new
    instance with suspicion_score filled in.

    Does NOT modify the input object.
    """
    f1 = _dwell_anomaly(features)
    f2 = _revisit_score(features)
    f3 = features.trajectory_irregularity          # already in [0, 1]
    f4 = 1.0 if features.billing_bypassed else 0.0

    raw = ALPHA * f1 + BETA * f2 + GAMMA * f3 + DELTA * f4
    score = max(0.0, min(1.0, raw))

    # Return a copy with the score attached.
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
# Sub-feature helpers
# ---------------------------------------------------------------------------

def _dwell_anomaly(features: BehaviorFeatures) -> float:
    """Max single-zone dwell time normalised to [0, 1]."""
    if not features.dwell_per_zone:
        return 0.0
    max_dwell = max(features.dwell_per_zone.values())
    return min(max_dwell / _DWELL_ANOMALY_THRESHOLD_S, 1.0)


def _revisit_score(features: BehaviorFeatures) -> float:
    """Total zone re-entries normalised to [0, 1]."""
    total_revisits = sum(features.zone_revisits.values())
    return min(total_revisits / _REVISIT_SATURATION, 1.0)
