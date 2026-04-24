"""Unit tests for src/zone_graph/zone_mapper.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "store_layout.yaml")

from src.zone_graph.zone_mapper import ZoneMapper


def _mapper():
    return ZoneMapper(CONFIG)


def test_entrance_zone():
    # Foot position at (192, 260) — inside entrance polygon [0,220]→[384,288]
    m = _mapper()
    assert m.get_zone((172, 220, 40, 40)) == "entrance"   # foot=(192, 260)


def test_walkway_zone():
    # Foot at (192, 200) — inside walkway [0,150]→[384,220]
    m = _mapper()
    assert m.get_zone((172, 160, 40, 40)) == "walkway"    # foot=(192, 200)


def test_shelves_left_zone():
    # Foot at (65, 120) — inside shelves_left [0,60]→[130,150]
    m = _mapper()
    assert m.get_zone((45, 80, 40, 40)) == "shelves_left"  # foot=(65, 120)


def test_shelves_center_zone():
    # Foot at (195, 120) — inside shelves_center [130,60]→[260,150]
    m = _mapper()
    assert m.get_zone((175, 80, 40, 40)) == "shelves_center"


def test_shelves_right_zone():
    # Foot at (322, 120) — inside shelves_right [260,60]→[384,150]
    m = _mapper()
    assert m.get_zone((302, 80, 40, 40)) == "shelves_right"


def test_billing_zone():
    # Foot at (65, 40) — inside billing [0,0]→[130,60]
    m = _mapper()
    assert m.get_zone((45, 0, 40, 40)) == "billing"        # foot=(65, 40)


def test_exit_zone():
    # Foot at (322, 40) — inside exit [260,0]→[384,60]
    m = _mapper()
    assert m.get_zone((302, 0, 40, 40)) == "exit"


def test_unknown_zone():
    # Foot at (195, 55) — in the gap between shelves and billing/exit rows
    m = _mapper()
    # Gap: x=130..260, y=0..60 (between billing and exit) — not covered by any polygon
    assert m.get_zone((175, 20, 40, 20)) == "unknown"      # foot=(195, 40) — in billing/exit gap


def test_zone_names_complete():
    m = _mapper()
    expected = {"entrance", "walkway", "shelves_left", "shelves_center",
                "shelves_right", "billing", "exit"}
    assert set(m.zone_names()) == expected


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
