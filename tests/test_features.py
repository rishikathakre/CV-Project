"""Unit tests for src/behavior/features.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_types import TrackedPerson
from src.behavior.features import BehaviorTracker, _ZONE_DEBOUNCE_FRAMES
from src.zone_graph.graph import ZoneTransitionGraph

# Repeat a zone this many times so it passes the debounce threshold.
_D = _ZONE_DEBOUNCE_FRAMES


def _person(pid: int, zone: str, frame: int = 0) -> TrackedPerson:
    return TrackedPerson(id=pid, frame=frame, timestamp=frame / 25.0,
                         bbox=(10, 10, 40, 80), zone=zone)


def _feed(tracker, graph, pid: int, zones: list, reps_each: int = _D) -> None:
    """Feed each zone `reps_each` times so the debounce commits each transition."""
    frame = 0
    for zone in zones:
        for _ in range(reps_each):
            tracker.update(_person(pid, zone, frame), graph)
            frame += 1


def _setup():
    return BehaviorTracker(fps=25.0), ZoneTransitionGraph()


def test_dwell_accumulates():
    tracker, graph = _setup()
    for f in range(25):
        tracker.update(_person(1, "entrance", f), graph)
    feat = tracker.get_features(1)
    assert abs(feat.dwell_per_zone["entrance"] - 1.0) < 0.01


def test_zone_sequence_appends_on_change():
    tracker, graph = _setup()
    _feed(tracker, graph, 1, ["entrance", "walkway", "shelves_left"])
    feat = tracker.get_features(1)
    assert feat.zone_sequence == ["entrance", "walkway", "shelves_left"]


def test_zone_sequence_no_duplicate_consecutive():
    tracker, graph = _setup()
    for _ in range(_D * 2):
        tracker.update(_person(1, "walkway"), graph)
    feat = tracker.get_features(1)
    assert feat.zone_sequence == ["walkway"]


def test_revisit_counted_on_reentry():
    tracker, graph = _setup()
    _feed(tracker, graph, 1, ["entrance", "walkway", "entrance"])
    feat = tracker.get_features(1)
    assert feat.zone_revisits.get("entrance", 0) == 1


def test_no_revisit_on_first_entry():
    tracker, graph = _setup()
    _feed(tracker, graph, 1, ["walkway"])
    feat = tracker.get_features(1)
    assert feat.zone_revisits == {}


def test_billing_bypassed_true():
    tracker, graph = _setup()
    _feed(tracker, graph, 1, ["entrance", "shelves", "exit"])
    feat = tracker.get_features(1)
    assert feat.billing_bypassed is True


def test_billing_bypassed_false_when_billing_visited():
    tracker, graph = _setup()
    _feed(tracker, graph, 1, ["entrance", "billing", "shelves", "exit"])
    feat = tracker.get_features(1)
    assert feat.billing_bypassed is False


def test_billing_bypass_requires_shelf_visit():
    """Walking straight entrance→exit with no shelf visit should NOT flag bypass."""
    tracker, graph = _setup()
    _feed(tracker, graph, 1, ["entrance", "walkway", "exit"])
    feat = tracker.get_features(1)
    assert feat.billing_bypassed is False


def test_custom_zone_names_shelf_match():
    """Zone named 'shelves' (not shelves_left/center/right) must still trigger bypass."""
    tracker, graph = _setup()
    _feed(tracker, graph, 1, ["entrance", "shelves", "exit"])
    feat = tracker.get_features(1)
    assert feat.billing_bypassed is True


def test_graph_records_transitions():
    tracker, graph = _setup()
    _feed(tracker, graph, 1, ["entrance", "walkway", "shelves_left"])
    assert graph.transition_count("entrance", "walkway") == 1
    assert graph.transition_count("walkway", "shelves_left") == 1


def test_multiple_persons_independent():
    tracker, graph = _setup()
    _feed(tracker, graph, 1, ["entrance"])
    _feed(tracker, graph, 2, ["exit"])
    assert tracker.get_features(1).zone_sequence == ["entrance"]
    assert tracker.get_features(2).zone_sequence == ["exit"]


def test_trajectory_irregularity_zero_single_zone():
    tracker, graph = _setup()
    for _ in range(_D * 2):
        tracker.update(_person(1, "walkway"), graph)
    feat = tracker.get_features(1)
    assert feat.trajectory_irregularity == 0.0


def test_trajectory_irregularity_elevated_with_shelf_bouncing():
    tracker, graph = _setup()
    _feed(tracker, graph, 1,
          ["shelves_left", "shelves_center", "shelves_left", "shelves_right"])
    feat = tracker.get_features(1)
    assert feat.trajectory_irregularity > 0.0


def test_debounce_suppresses_single_frame_oscillation():
    """One-frame flip to a new zone should NOT count as a zone transition."""
    tracker, graph = _setup()
    # Stay in entrance for a long time, briefly touch walkway, come back
    for _ in range(_D * 3):
        tracker.update(_person(1, "entrance"), graph)
    # Single frame in walkway — below debounce threshold
    tracker.update(_person(1, "walkway"), graph)
    for _ in range(_D * 3):
        tracker.update(_person(1, "entrance"), graph)
    feat = tracker.get_features(1)
    # walkway should not appear in zone_sequence
    assert "walkway" not in feat.zone_sequence
    # no revisit of entrance from the brief walkway flicker
    assert feat.zone_revisits.get("entrance", 0) == 0


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
