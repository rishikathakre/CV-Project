from typing import Dict, List, Optional

from shared.data_types import BehaviorFeatures, TrackedPerson
from src.zone_graph.graph import ZoneTransitionGraph


def _is_shelf_zone(zone: str) -> bool:
    canonical = {"shelves_left", "shelves_center", "shelves_right"}
    return zone in canonical or zone.startswith("shelf") or zone.startswith("shelves")


def _is_billing_zone(zone: str) -> bool:
    return zone == "billing" or zone.startswith("billing") or zone.startswith("checkout")


def _is_exit_zone(zone: str) -> bool:
    return zone == "exit" or zone.startswith("exit")


# A person must stay in the same zone for this many frames before we count it.
# This stops brief mis-detections from being recorded as real zone visits.
_ZONE_DEBOUNCE_FRAMES = 8

# A person must spend at least this many frames near shelves before we flag
# them for billing bypass. This filters out people who just walked past.
_MIN_SHELF_FRAMES_FOR_BYPASS = 500


class _PersonState:
    """Stores the running behavioral state for one tracked person."""

    def __init__(self, person_id: int, fps: float) -> None:
        self.id  = person_id
        self.fps = fps
        self.dwell_frames: Dict[str, int] = {}
        self.zone_revisits: Dict[str, int] = {}
        self.zone_sequence: List[str] = []
        self.visited_zones: set = set()
        self.billing_seen:       bool = False
        self.billing_bypassed:   bool = False
        self.shelf_visited:      bool = False
        self.shelf_dwell_frames: int  = 0
        self.last_zone: Optional[str] = None
        self._candidate_zone:   Optional[str] = None
        self._candidate_frames: int = 0

    def update(self, raw_zone: str) -> Optional[str]:
        """Process one new zone observation and return the previous zone if a transition happened."""
        # Only commit to a new zone after staying there for enough frames.
        if raw_zone == self._candidate_zone:
            self._candidate_frames += 1
        else:
            self._candidate_zone   = raw_zone
            self._candidate_frames = 1

        if self._candidate_frames >= _ZONE_DEBOUNCE_FRAMES:
            committed = raw_zone
        else:
            committed = self.last_zone if self.last_zone is not None else raw_zone

        self.dwell_frames[committed] = self.dwell_frames.get(committed, 0) + 1

        transition_from: Optional[str] = None
        if committed != self.last_zone:
            transition_from = self.last_zone
            if committed in self.visited_zones:
                self.zone_revisits[committed] = self.zone_revisits.get(committed, 0) + 1
            self.visited_zones.add(committed)
            self.zone_sequence.append(committed)
            self.last_zone = committed

        if _is_shelf_zone(committed):
            self.shelf_visited = True
            self.shelf_dwell_frames += 1
        if _is_billing_zone(committed):
            self.billing_seen = True

        # Flag billing bypass only when the person spent enough time at shelves
        # and then left via the exit without ever going to billing.
        if (_is_exit_zone(committed)
                and self.shelf_visited
                and not self.billing_seen
                and self.shelf_dwell_frames >= _MIN_SHELF_FRAMES_FOR_BYPASS):
            self.billing_bypassed = True

        return transition_from

    def build_features(self) -> BehaviorFeatures:
        """Build a BehaviorFeatures snapshot from the current state."""
        dwell_seconds = {z: frames / self.fps for z, frames in self.dwell_frames.items()}
        irregularity  = _compute_trajectory_irregularity(self.zone_sequence)
        return BehaviorFeatures(
            id=self.id,
            dwell_per_zone=dict(dwell_seconds),
            zone_revisits=dict(self.zone_revisits),
            zone_sequence=list(self.zone_sequence),
            billing_bypassed=self.billing_bypassed,
            trajectory_irregularity=irregularity,
            suspicion_score=0.0,
            alert_reasons=[],
        )


def _compute_trajectory_irregularity(zone_sequence: List[str]) -> float:
    """Return a score between 0 and 1 for how erratic the person's path is.

    We count how many consecutive shelf-to-shelf transitions there are
    (jumping between different shelf zones without going to the walkway).
    More of these relative to total transitions means a higher score.
    """
    if len(zone_sequence) < 2:
        return 0.0
    transitions = list(zip(zone_sequence, zone_sequence[1:]))
    total = len(transitions)
    irregular = sum(1 for a, b in transitions if _is_shelf_zone(a) and _is_shelf_zone(b) and a != b)
    return min(irregular / total, 1.0)


class BehaviorTracker:
    """Keeps a separate _PersonState for every tracked person and updates it each frame."""

    def __init__(self, fps: float = 25.0) -> None:
        self._fps    = fps
        self._states: Dict[int, _PersonState] = {}

    def update(self, person: TrackedPerson, zone_graph: ZoneTransitionGraph) -> BehaviorFeatures:
        """Feed one person observation and return their latest behavioral features."""
        pid  = person.id
        zone = person.zone
        if pid not in self._states:
            self._states[pid] = _PersonState(pid, self._fps)
        state     = self._states[pid]
        prev_zone = state.update(zone)
        if prev_zone is not None:
            zone_graph.add_transition(prev_zone, zone)
        return state.build_features()

    def get_features(self, person_id: int) -> Optional[BehaviorFeatures]:
        """Return the current features for one person, or None if they have not been seen."""
        state = self._states.get(person_id)
        return state.build_features() if state else None

    def all_features(self) -> Dict[int, BehaviorFeatures]:
        """Return the current features for every tracked person."""
        return {pid: state.build_features() for pid, state in self._states.items()}

    def tracked_ids(self) -> List[int]:
        return list(self._states.keys())
