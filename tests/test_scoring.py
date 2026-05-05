import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_types import BehaviorFeatures
from src.behavior.scoring import (
    ALPHA, BETA, DELTA, GAMMA,
    _DWELL_ANOMALY_THRESHOLD_S, _REVISIT_SATURATION,
    compute_score,
)


def _feat(**kwargs) -> BehaviorFeatures:
    defaults = dict(
        id=1,
        dwell_per_zone={},
        zone_revisits={},
        zone_sequence=[],
        billing_bypassed=False,
        trajectory_irregularity=0.0,
        suspicion_score=0.0,
        alert_reasons=[],
    )
    defaults.update(kwargs)
    return BehaviorFeatures(**defaults)


def test_zero_score_for_clean_person():
    feat = _feat()
    result = compute_score(feat)
    assert result.suspicion_score == 0.0


def test_billing_bypass_contributes_delta():
    feat = _feat(billing_bypassed=True)
    result = compute_score(feat)
    assert abs(result.suspicion_score - DELTA) < 1e-6


def test_full_dwell_contributes_alpha():
    feat = _feat(dwell_per_zone={"shelves_left": _DWELL_ANOMALY_THRESHOLD_S})
    result = compute_score(feat)
    assert abs(result.suspicion_score - ALPHA) < 1e-6


def test_partial_dwell_scales():
    half = _DWELL_ANOMALY_THRESHOLD_S / 2
    feat = _feat(dwell_per_zone={"walkway": half})
    result = compute_score(feat)
    assert abs(result.suspicion_score - ALPHA * 0.5) < 1e-6


def test_revisit_saturation():
    feat = _feat(zone_revisits={"shelves_left": _REVISIT_SATURATION})
    result = compute_score(feat)
    assert abs(result.suspicion_score - BETA) < 1e-6


def test_trajectory_irregularity_scales():
    feat = _feat(trajectory_irregularity=0.5)
    result = compute_score(feat)
    assert abs(result.suspicion_score - GAMMA * 0.5) < 1e-6


def test_score_clamped_to_one():
    feat = _feat(
        dwell_per_zone={"shelves_left": _DWELL_ANOMALY_THRESHOLD_S},
        zone_revisits={"x": _REVISIT_SATURATION},
        trajectory_irregularity=1.0,
        billing_bypassed=True,
    )
    result = compute_score(feat)
    assert result.suspicion_score <= 1.0


def test_score_never_negative():
    feat = _feat(trajectory_irregularity=-0.5)
    result = compute_score(feat)
    assert result.suspicion_score >= 0.0


def test_input_object_not_mutated():
    feat = _feat(billing_bypassed=True)
    original_score = feat.suspicion_score
    compute_score(feat)
    assert feat.suspicion_score == original_score


def test_combined_score_additive():
    feat = _feat(
        dwell_per_zone={"shelves_left": _DWELL_ANOMALY_THRESHOLD_S},
        billing_bypassed=True,
    )
    result = compute_score(feat)
    expected = ALPHA * 1.0 + DELTA * 1.0
    assert abs(result.suspicion_score - expected) < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
