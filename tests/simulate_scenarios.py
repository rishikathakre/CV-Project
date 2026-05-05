import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_types import TrackedPerson
from src.behavior.features import BehaviorTracker, _ZONE_DEBOUNCE_FRAMES
from src.behavior.scoring import AdaptiveScorer
from src.alerts.explainer import generate_alert, get_alert_level
from src.zone_graph.graph import ZoneTransitionGraph

FPS = 25.0
_D  = _ZONE_DEBOUNCE_FRAMES

_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

_LEVEL_COLOR = {
    "HIGH":   _RED,
    "MEDIUM": _YELLOW,
    "LOW":    _CYAN,
    "NONE":   _GREEN,
}

SCENARIOS = [
    {
        "name": "Normal shopper - visits billing before leaving",
        "path": [
            ("entrance",  3),
            ("walkway",   4),
            ("shelves",  20),
            ("billing",   8),
            ("exit",      2),
        ],
    },
    {
        "name": "Billing bypass - shelves -> exit, skips billing",
        "path": [
            ("entrance",  3),
            ("walkway",   4),
            ("shelves",  30),
            ("exit",      2),
        ],
    },
    {
        "name": "Shelf loiterer - lingers at shelves 90+ seconds",
        "path": [
            ("entrance",  3),
            ("shelves",  95),
            ("billing",   5),
            ("exit",      2),
        ],
    },
    {
        "name": "Zone revisitor - re-enters shelves 3 times",
        "path": [
            ("entrance",  3),
            ("shelves",  15),
            ("walkway",   4),
            ("shelves",  15),
            ("walkway",   4),
            ("shelves",  15),
            ("exit",      2),
        ],
    },
    {
        "name": "Suspicious - multiple revisits + billing bypass",
        "path": [
            ("entrance",  3),
            ("shelves",  20),
            ("walkway",   3),
            ("shelves",  20),
            ("walkway",   3),
            ("shelves",  20),
            ("exit",      2),
        ],
    },
    {
        "name": "Walk-through - just passing, no shelves",
        "path": [
            ("entrance",  2),
            ("walkway",  10),
            ("exit",      2),
        ],
    },
    {
        "name": "Window shopper - brief shelf visit, leaves via billing",
        "path": [
            ("entrance",  3),
            ("walkway",   5),
            ("shelves",  10),
            ("walkway",   3),
            ("billing",   2),
            ("exit",      2),
        ],
    },
    {
        "name": "Erratic movement - shelf bouncing back and forth",
        "path": [
            ("entrance",       3),
            ("shelves_left",   8),
            ("shelves_right",  8),
            ("shelves_left",   8),
            ("shelves_center", 8),
            ("shelves_left",   8),
            ("exit",           2),
        ],
    },
]


def _make_person(pid: int, zone: str, frame: int) -> TrackedPerson:
    return TrackedPerson(id=pid, frame=frame, timestamp=frame / FPS,
                         bbox=(100, 100, 60, 160), zone=zone)


def _run_scenario(pid: int, path: list, tracker: BehaviorTracker,
                  graph: ZoneTransitionGraph, scorer: AdaptiveScorer) -> dict:
    frame = 0
    for zone, seconds in path:
        n_frames = max(int(seconds * FPS), _D)
        for _ in range(n_frames):
            person   = _make_person(pid, zone, frame)
            features = tracker.update(person, graph)
            features, _ = scorer.compute(features)
            frame += 1

    features = tracker.get_features(pid)
    features, breakdown = scorer.compute_final(features)
    features = generate_alert(features)
    level    = get_alert_level(features)

    return {
        "level":     level,
        "score":     features.suspicion_score,
        "breakdown": breakdown,
        "reasons":   features.alert_reasons,
        "zones":     features.zone_sequence,
        "revisits":  sum(features.zone_revisits.values()),
        "bypass":    features.billing_bypassed,
        "dwell":     {z: round(t, 1) for z, t in features.dwell_per_zone.items()},
    }


def main():
    print(f"\n{'='*70}")
    print(f"  {_BOLD}Retail Risk Monitor - Scenario Simulator{_RESET}")
    print(f"{'='*70}\n")

    scorer  = AdaptiveScorer()
    graph   = ZoneTransitionGraph()
    tracker = BehaviorTracker(fps=FPS)

    results = []
    for pid, scenario in enumerate(SCENARIOS, start=1):
        r = _run_scenario(pid, scenario["path"], tracker, graph, scorer)
        results.append((scenario["name"], r))

    col_w = 52
    print(f"  {'Scenario':<{col_w}}  {'Alert':<8}  {'Score':>6}  {'Bypass':>8}  {'Revisits':>9}")
    print(f"  {'-'*col_w}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*9}")

    for name, r in results:
        color = _LEVEL_COLOR.get(r["level"], "")
        bypass_str = "YES (!)" if r["bypass"] else "no"
        print(
            f"  {name:<{col_w}}  "
            f"{color}{r['level']:<8}{_RESET}  "
            f"{r['score']:>6.3f}  "
            f"{bypass_str:>8}  "
            f"{r['revisits']:>9}"
        )

    print(f"\n{'='*70}")
    print(f"  {_BOLD}Detailed breakdown{_RESET}")
    print(f"{'='*70}")

    for name, r in results:
        color = _LEVEL_COLOR.get(r["level"], "")
        print(f"\n  {_BOLD}{name}{_RESET}")
        print(f"    Alert   : {color}{r['level']}{_RESET}  (score {r['score']:.3f})")
        print(f"    Zones   : {' -> '.join(r['zones'])}")
        print(f"    Dwell   : {r['dwell']}")
        bk = r["breakdown"]
        print(
            f"    Score breakdown: "
            f"dwell {bk.get('Dwell anomaly',0):+d}  "
            f"revisits {bk.get('Zone revisits',0):+d}  "
            f"irregular {bk.get('Path irregularity',0):+d}  "
            f"bypass {bk.get('Billing bypass',0):+d}"
        )
        for reason in r["reasons"]:
            print(f"      * {reason}")

    print(f"\n{'='*70}\n")

    th = scorer.current_thresholds()
    mode = "Adaptive" if th["adaptive"] else "Fixed fallback"
    print(f"  Scorer mode    : {mode}  (n={th['n']} persons)")
    print(f"  Dwell threshold: {th['dwell_s']} s")
    print(f"  Revisit saturation: {th['revisits']}\n")


if __name__ == "__main__":
    main()
