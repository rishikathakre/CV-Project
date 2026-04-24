"""Unit tests for src/behavior/features.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_types import TrackedPerson
from src.behavior.features import BehaviorTracker
from src.zone_graph.graph import ZoneTransitionGraph


def _person(pid: int, zone: str, frame: int = 0) -> TrackedPerson:
    return TrackedPerson(id=pid, frame=frame, timestamp=frame / 25.0,
                         bbox=(10, 10, 40, 80), zone=zone)


def _setup():
    graph = ZoneTransitionGraph()
    tracker = BehaviorTracker(fps=25.0)
    return tracker, graph


def test_dwell_accumulates():
    tracker, graph = _setup()
    for f in range(25):   # 1 second at 25 fps
        tracker.update(_person(1, "entrance", f), graph)
    feat = tracker.get_features(1)
    assert abs(feat.dwell_per_zone["entrance"] - 1.0) < 0.01


def test_zone_sequence_appends_on_change():
    tracker, graph = _setup()
    for zone in ["entrance", "walkway", "shelves_left"]:
        tracker.update(_person(1, zone), graph)
    feat = tracker.get_features(1)
    assert feat.zone_sequence == ["entrance", "walkway", "shelves_left"]


def test_zone_sequence_no_duplicate_consecutive():
    tracker, graph = _setup()
    # Same zone repeated — should appear once in sequence
    for _ in range(5):
        tracker.update(_person(1, "walkway"), graph)
    feat = tracker.get_features(1)
    assert feat.zone_sequence == ["walkway"]


def test_revisit_counted_on_reentry():
    tracker, graph = _setup()
    for zone in ["entrance", "walkway", "entrance"]:
        tracker.update(_person(1, zone), graph)
    feat = tracker.get_features(1)
    assert feat.zone_revisits.get("entrance", 0) == 1


def test_no_revisit_on_first_entry():
    tracker, graph = _setup()
    tracker.update(_person(1, "walkway"), graph)
    feat = tracker.get_features(1)
    assert feat.zone_revisits == {}


def test_billing_bypassed_true():
    tracker, graph = _setup()
    for zone in ["entrance", "walkway", "exit"]:
        tracker.update(_person(1, zone), graph)
    feat = tracker.get_features(1)
    assert feat.billing_bypassed is True


def test_billing_bypassed_false_when_billing_visited():
    tracker, graph = _setup()
    for zone in ["entrance", "billing", "exit"]:
        tracker.update(_person(1, zone), graph)
    feat = tracker.get_features(1)
    assert feat.billing_bypassed is False


def test_graph_records_transitions():
    tracker, graph = _setup()
    for zone in ["entrance", "walkway", "shelves_left"]:
        tracker.update(_person(1, zone), graph)
    assert graph.transition_count("entrance", "walkway") == 1
    assert graph.transition_count("walkway", "shelves_left") == 1


def test_multiple_persons_independent():
    tracker, graph = _setup()
    tracker.update(_person(1, "entrance"), graph)
    tracker.update(_person(2, "exit"), graph)
    assert tracker.get_features(1).zone_sequence == ["entrance"]
    assert tracker.get_features(2).zone_sequence == ["exit"]


def test_trajectory_irregularity_zero_single_zone():
    tracker, graph = _setup()
    for _ in range(5):
        tracker.update(_person(1, "walkway"), graph)
    feat = tracker.get_features(1)
    assert feat.trajectory_irregularity == 0.0


def test_trajectory_irregularity_elevated_with_shelf_bouncing():
    tracker, graph = _setup()
    for zone in ["shelves_left", "shelves_center", "shelves_left", "shelves_right"]:
        tracker.update(_person(1, zone), graph)
    feat = tracker.get_features(1)
    assert feat.trajectory_irregularity > 0.0


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
