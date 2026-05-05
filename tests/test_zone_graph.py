import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.zone_graph.graph import ZoneTransitionGraph


def test_add_and_query_transition():
    g = ZoneTransitionGraph()
    g.add_transition("entrance", "walkway")
    g.add_transition("entrance", "walkway")
    assert g.transition_count("entrance", "walkway") == 2


def test_self_loop_ignored():
    g = ZoneTransitionGraph()
    g.add_transition("walkway", "walkway")
    assert g.edge_count() == 0


def test_missing_edge_returns_zero():
    g = ZoneTransitionGraph()
    assert g.transition_count("shelves_left", "billing") == 0


def test_out_edges():
    g = ZoneTransitionGraph()
    g.add_transition("walkway", "shelves_left")
    g.add_transition("walkway", "shelves_right")
    g.add_transition("walkway", "shelves_right")
    edges = g.out_edges("walkway")
    assert edges["shelves_left"] == 1
    assert edges["shelves_right"] == 2


def test_most_common_transition():
    g = ZoneTransitionGraph()
    g.add_transition("walkway", "shelves_left")
    g.add_transition("walkway", "shelves_center")
    g.add_transition("walkway", "shelves_center")
    best, count = g.most_common_transition("walkway")
    assert best == "shelves_center"
    assert count == 2


def test_most_common_transition_unknown_zone():
    g = ZoneTransitionGraph()
    assert g.most_common_transition("nonexistent") is None


def test_all_edges():
    g = ZoneTransitionGraph()
    g.add_transition("a", "b")
    g.add_transition("b", "c")
    edges = g.all_edges()
    assert ("a", "b") in edges
    assert ("b", "c") in edges
    assert len(edges) == 2


def test_node_and_edge_counts():
    g = ZoneTransitionGraph()
    g.add_transition("entrance", "walkway")
    g.add_transition("walkway", "shelves_left")
    assert g.node_count() == 3
    assert g.edge_count() == 2


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
