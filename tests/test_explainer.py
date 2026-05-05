import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_types import BehaviorFeatures
from src.alerts.explainer import generate_alert, get_alert_level


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


def test_none_level_for_clean_person():
    assert get_alert_level(_feat()) == "NONE"


def test_high_level_billing_bypass():
    assert get_alert_level(_feat(billing_bypassed=True)) == "HIGH"


def test_high_level_three_revisits():
    feat = _feat(zone_revisits={"shelves_left": 3})
    assert get_alert_level(feat) == "HIGH"


def test_high_level_score_threshold():
    assert get_alert_level(_feat(suspicion_score=0.75)) == "HIGH"


def test_medium_level_shelf_dwell():
    feat = _feat(dwell_per_zone={"shelves_center": 35.0})
    assert get_alert_level(feat) == "MEDIUM"


def test_medium_level_score_threshold():
    assert get_alert_level(_feat(suspicion_score=0.45)) == "MEDIUM"


def test_low_level_small_score():
    assert get_alert_level(_feat(suspicion_score=0.1)) == "LOW"


def test_billing_bypass_reason_in_output():
    feat = generate_alert(_feat(billing_bypassed=True))
    assert any("billing" in r.lower() for r in feat.alert_reasons)


def test_high_revisit_reason_included():
    feat = generate_alert(_feat(zone_revisits={"shelves_left": 4}))
    reasons_text = " ".join(feat.alert_reasons).lower()
    assert "shelves_left" in reasons_text


def test_shelf_dwell_reason_included():
    feat = generate_alert(_feat(dwell_per_zone={"shelves_right": 40.0}))
    reasons_text = " ".join(feat.alert_reasons).lower()
    assert "shelves_right" in reasons_text


def test_alert_level_badge_first_reason():
    feat = generate_alert(_feat(billing_bypassed=True))
    assert feat.alert_reasons[0].startswith("Alert level: HIGH")


def test_none_level_produces_no_reasons():
    feat = generate_alert(_feat())
    assert feat.alert_reasons == []


def test_input_object_not_mutated():
    original = _feat(billing_bypassed=True)
    original_reasons = list(original.alert_reasons)
    generate_alert(original)
    assert original.alert_reasons == original_reasons


def test_high_overrides_medium():
    # billing bypass (HIGH) + shelf dwell (MEDIUM) → must be HIGH
    feat = _feat(billing_bypassed=True, dwell_per_zone={"shelves_left": 40.0})
    assert get_alert_level(feat) == "HIGH"


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
