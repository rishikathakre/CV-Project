import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "store_layout.yaml")

from src.zone_graph.zone_mapper import ZoneMapper


def _mapper():
    return ZoneMapper(CONFIG)


def test_entrance_zone():
    m = _mapper()
    assert m.get_zone((172, 220, 40, 40)) == "entrance"


def test_walkway_zone():
    m = _mapper()
    assert m.get_zone((172, 160, 40, 40)) == "walkway"


def test_shelves_left_zone():
    m = _mapper()
    assert m.get_zone((45, 80, 40, 40)) == "shelves_left"


def test_shelves_center_zone():
    m = _mapper()
    assert m.get_zone((175, 80, 40, 40)) == "shelves_center"


def test_shelves_right_zone():
    m = _mapper()
    assert m.get_zone((302, 80, 40, 40)) == "shelves_right"


def test_billing_zone():
    m = _mapper()
    assert m.get_zone((45, 0, 40, 40)) == "billing"


def test_exit_zone():
    m = _mapper()
    assert m.get_zone((302, 0, 40, 40)) == "exit"


def test_unknown_zone():
    m = _mapper()
    assert m.get_zone((175, 20, 40, 20)) == "unknown"


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
