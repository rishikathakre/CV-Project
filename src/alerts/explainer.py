from typing import List, Tuple

from shared.data_types import BehaviorFeatures

# Score thresholds for the three alert levels.
_HIGH_SCORE_THRESHOLD   = 0.7
_MEDIUM_SCORE_THRESHOLD = 0.5

# How long someone must stay near a shelf before we raise a MEDIUM alert.
_SHELF_DWELL_MEDIUM_S   = 60.0

# Number of revisits to the same zone that triggers a HIGH alert.
_HIGH_REVISIT_COUNT     = 2

# These are the canonical shelf zone names used in the YAML config.
_SHELF_ZONES_CANONICAL = {"shelves_left", "shelves_center", "shelves_right"}


def _is_shelf_zone(zone: str) -> bool:
    return zone in _SHELF_ZONES_CANONICAL or zone.startswith("shelf") or zone.startswith("shelves")


def generate_alert(features: BehaviorFeatures) -> BehaviorFeatures:
    """Evaluate the features, pick an alert level, and return updated features with reasons."""
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
    """Return only the alert level string without building the full alert."""
    level, _ = _evaluate(features)
    return level


def _evaluate(features: BehaviorFeatures) -> Tuple[str, List[str]]:
    """Apply all rules in priority order and return the level and list of reasons.

    Rules are checked in this order: HIGH rules first, then MEDIUM, then LOW.
    Once a HIGH is triggered no MEDIUM rules are checked.
    """
    reasons: List[str] = []
    level = "NONE"

    # Rule 1: billing bypass is an immediate HIGH.
    if features.billing_bypassed:
        level = "HIGH"
        reasons.append("Left via exit without visiting the billing counter.")

    # Rule 2: coming back to the same zone multiple times is suspicious.
    high_revisit_zones = [z for z, count in features.zone_revisits.items() if count >= _HIGH_REVISIT_COUNT]
    if high_revisit_zones:
        level = "HIGH"
        for z in high_revisit_zones:
            reasons.append(
                f"Re-entered '{z}' {features.zone_revisits[z]} times "
                f"(threshold: {_HIGH_REVISIT_COUNT})."
            )

    # Rule 3: a very high composite score also triggers HIGH.
    if features.suspicion_score >= _HIGH_SCORE_THRESHOLD:
        level = "HIGH"
        reasons.append(
            f"Overall suspicion score {features.suspicion_score:.2f} "
            f"exceeds high threshold ({_HIGH_SCORE_THRESHOLD})."
        )

    # MEDIUM rules only apply if no HIGH was triggered.
    if level != "HIGH":
        # Rule 4: lingering near shelves for too long.
        shelf_dwell_flags = [
            (z, t) for z, t in features.dwell_per_zone.items()
            if _is_shelf_zone(z) and t > _SHELF_DWELL_MEDIUM_S
        ]
        if shelf_dwell_flags:
            level = "MEDIUM"
            for z, t in shelf_dwell_flags:
                reasons.append(f"Spent {t:.1f}s in '{z}' (threshold: {_SHELF_DWELL_MEDIUM_S:.0f}s).")

        # Rule 5: medium composite score.
        if features.suspicion_score >= _MEDIUM_SCORE_THRESHOLD:
            if level != "MEDIUM":
                level = "MEDIUM"
            reasons.append(
                f"Overall suspicion score {features.suspicion_score:.2f} "
                f"exceeds medium threshold ({_MEDIUM_SCORE_THRESHOLD})."
            )

    # Any non-zero score that hasn't triggered a higher alert gets LOW.
    if level == "NONE" and features.suspicion_score > 0.0:
        level = "LOW"
        reasons.append(f"Minor suspicious indicators detected (score: {features.suspicion_score:.2f}).")

    # Extra context added to any alert that shows irregular movement.
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
