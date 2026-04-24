"""
Natural-language alert generation.

Translates a BehaviorFeatures snapshot into an alert level (LOW / MEDIUM / HIGH)
and a list of plain-English reason strings that explain why the score is elevated.

Alert level rules (applied in priority order — first match wins for level):
  HIGH   : billing_bypassed is True
  HIGH   : any zone has 3+ revisits
  HIGH   : suspicion_score >= 0.7
  MEDIUM : any shelf zone dwell > 30 seconds
  MEDIUM : suspicion_score >= 0.4
  LOW    : everything else (score > 0 but nothing specific triggered)
  NONE   : suspicion_score == 0.0 and no flags
"""

from typing import List, Tuple

from shared.data_types import BehaviorFeatures

# Thresholds
_HIGH_SCORE_THRESHOLD   = 0.7
_MEDIUM_SCORE_THRESHOLD = 0.4
_SHELF_DWELL_MEDIUM_S   = 30.0   # seconds
_HIGH_REVISIT_COUNT     = 3

_SHELF_ZONES = {"shelves_left", "shelves_center", "shelves_right"}


def generate_alert(features: BehaviorFeatures) -> BehaviorFeatures:
    """
    Compute alert level and reason strings; return a new BehaviorFeatures
    instance with alert_reasons populated.

    The caller is expected to have already run scoring.compute_score() so
    that features.suspicion_score is set.
    """
    level, reasons = _evaluate(features)

    if level != "NONE":
        reasons.insert(0, f"Alert level: {level}")

    return BehaviorFeatures(
        id=features.id,
        dwell_per_zone=features.dwell_per_zone,
        zone_revisits=features.zone_revisits,
        zone_sequence=features.zone_sequence,
        billing_bypassed=features.billing_bypassed,
        trajectory_irregularity=features.trajectory_irregularity,
        suspicion_score=features.suspicion_score,
        alert_reasons=reasons,
    )


def get_alert_level(features: BehaviorFeatures) -> str:
    """Return just the alert level string without building a new object."""
    level, _ = _evaluate(features)
    return level


# ---------------------------------------------------------------------------
# Internal logic
# ---------------------------------------------------------------------------

def _evaluate(features: BehaviorFeatures) -> Tuple[str, List[str]]:
    """Return (level, reasons) without mutating features."""
    reasons: List[str] = []
    level = "NONE"

    # --- HIGH triggers ---
    if features.billing_bypassed:
        level = "HIGH"
        reasons.append("Left via exit without visiting the billing counter.")

    high_revisit_zones = [
        z for z, count in features.zone_revisits.items()
        if count >= _HIGH_REVISIT_COUNT
    ]
    if high_revisit_zones:
        level = "HIGH"
        for z in high_revisit_zones:
            reasons.append(
                f"Re-entered '{z}' {features.zone_revisits[z]} times "
                f"(threshold: {_HIGH_REVISIT_COUNT})."
            )

    if features.suspicion_score >= _HIGH_SCORE_THRESHOLD:
        level = "HIGH"
        reasons.append(
            f"Overall suspicion score {features.suspicion_score:.2f} "
            f"exceeds high threshold ({_HIGH_SCORE_THRESHOLD})."
        )

    # --- MEDIUM triggers (only if not already HIGH) ---
    if level != "HIGH":
        shelf_dwell_flags = [
            (z, t)
            for z, t in features.dwell_per_zone.items()
            if z in _SHELF_ZONES and t > _SHELF_DWELL_MEDIUM_S
        ]
        if shelf_dwell_flags:
            level = "MEDIUM"
            for z, t in shelf_dwell_flags:
                reasons.append(
                    f"Spent {t:.1f}s in '{z}' "
                    f"(threshold: {_SHELF_DWELL_MEDIUM_S:.0f}s)."
                )

        if features.suspicion_score >= _MEDIUM_SCORE_THRESHOLD:
            if level != "MEDIUM":   # don't duplicate if shelf dwell already set it
                level = "MEDIUM"
            reasons.append(
                f"Overall suspicion score {features.suspicion_score:.2f} "
                f"exceeds medium threshold ({_MEDIUM_SCORE_THRESHOLD})."
            )

    # --- LOW catch-all ---
    if level == "NONE" and features.suspicion_score > 0.0:
        level = "LOW"
        reasons.append(
            f"Minor suspicious indicators detected "
            f"(score: {features.suspicion_score:.2f})."
        )

    # --- Supplementary context reasons (all levels) ---
    if features.trajectory_irregularity > 0.3:
        reasons.append(
            f"Irregular movement pattern between shelf zones "
            f"(irregularity: {features.trajectory_irregularity:.2f})."
        )

    if len(features.zone_sequence) > 2:
        reasons.append(
            f"Visited {len(set(features.zone_sequence))} distinct zones: "
            + ", ".join(dict.fromkeys(features.zone_sequence)) + "."
        )

    return level, reasons
