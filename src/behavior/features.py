"""
Per-person behavioral feature extraction.

Maintains a running state dictionary keyed by tracking ID so that features
accumulate correctly across frames.  Call update() once per frame for each
TrackedPerson, then call get_features() to retrieve the current BehaviorFeatures.
"""

from typing import Dict, List, Optional

from shared.data_types import BehaviorFeatures, TrackedPerson
from src.zone_graph.graph import ZoneTransitionGraph


# ---------------------------------------------------------------------------
# Zone matching — flexible so user-drawn custom zones work
# ---------------------------------------------------------------------------

def _is_shelf_zone(zone: str) -> bool:
    """True for canonical shelf names AND any user zone containing 'shelf'/'shelves'."""
    canonical = {"shelves_left", "shelves_center", "shelves_right"}
    return zone in canonical or zone.startswith("shelf") or zone.startswith("shelves")


def _is_billing_zone(zone: str) -> bool:
    return zone == "billing" or zone.startswith("billing") or zone.startswith("checkout")


def _is_exit_zone(zone: str) -> bool:
    return zone == "exit" or zone.startswith("exit")


# ---------------------------------------------------------------------------
# Zone debounce: suppress false revisits from foot-position oscillation
# ---------------------------------------------------------------------------
# Foot position can oscillate between adjacent zone boundaries on every frame.
# Require this many consecutive frames in a new zone before committing the
# transition and counting it as a revisit.
_ZONE_DEBOUNCE_FRAMES = 8

# Billing bypass: only flag if the person spent this many frames in a shelf zone.
# Prevents false positives from people who just pass through the scene quickly.
_MIN_SHELF_FRAMES_FOR_BYPASS = 500   # ≈ 20 seconds at 25 fps


class _PersonState:
    """Internal mutable state for a single tracked person."""

    def __init__(self, person_id: int, fps: float) -> None:
        self.id  = person_id
        self.fps = fps

        # Dwell tracking: zone → frame count in that zone.
        self.dwell_frames: Dict[str, int] = {}

        # Revisit tracking: zone → number of re-entries (first entry not counted).
        self.zone_revisits: Dict[str, int] = {}

        # Ordered sequence of zone transitions (consecutive duplicates collapsed).
        self.zone_sequence: List[str] = []

        # Set of zones visited at all.
        self.visited_zones: set = set()

        # Billing-bypass flags.
        self.billing_seen:        bool = False
        self.billing_bypassed:    bool = False
        self.shelf_visited:       bool = False
        self.shelf_dwell_frames:  int  = 0

        # Last *committed* zone (after debounce).
        self.last_zone: Optional[str] = None

        # Debounce state.
        self._candidate_zone:   Optional[str] = None
        self._candidate_frames: int = 0

    # ------------------------------------------------------------------

    def update(self, raw_zone: str) -> Optional[str]:
        """
        Process one frame observation.

        Applies a debounce so that brief foot-position oscillations across a
        zone boundary don't inflate revisit counts.  Only commits a zone
        transition after _ZONE_DEBOUNCE_FRAMES consecutive readings in the
        new zone.

        Returns the previous committed zone if a transition was committed this
        frame (for the zone graph), else None.
        """
        # ---- Debounce ----
        if raw_zone == self._candidate_zone:
            self._candidate_frames += 1
        else:
            self._candidate_zone   = raw_zone
            self._candidate_frames = 1

        # Use the committed zone until debounce threshold is reached.
        if self._candidate_frames >= _ZONE_DEBOUNCE_FRAMES:
            committed = raw_zone
        else:
            committed = self.last_zone if self.last_zone is not None else raw_zone

        # ---- Dwell accumulation (against committed zone) ----
        self.dwell_frames[committed] = self.dwell_frames.get(committed, 0) + 1

        # ---- Zone transition / revisit ----
        transition_from: Optional[str] = None
        if committed != self.last_zone:
            transition_from = self.last_zone

            if committed in self.visited_zones:
                self.zone_revisits[committed] = self.zone_revisits.get(committed, 0) + 1
            self.visited_zones.add(committed)

            self.zone_sequence.append(committed)
            self.last_zone = committed

        # ---- Billing-bypass detection ----
        if _is_shelf_zone(committed):
            self.shelf_visited = True
            self.shelf_dwell_frames += 1
        if _is_billing_zone(committed):
            self.billing_seen = True
        if (_is_exit_zone(committed)
                and self.shelf_visited
                and not self.billing_seen
                and self.shelf_dwell_frames >= _MIN_SHELF_FRAMES_FOR_BYPASS):
            self.billing_bypassed = True

        return transition_from

    def build_features(self) -> BehaviorFeatures:
        """Snapshot the current state as a BehaviorFeatures instance."""
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
    """
    Heuristic: ratio of shelf ↔ shelf back-and-forth transitions to total.
    Returns [0.0, 1.0].
    """
    if len(zone_sequence) < 2:
        return 0.0
    transitions = list(zip(zone_sequence, zone_sequence[1:]))
    total = len(transitions)
    irregular = sum(
        1 for a, b in transitions
        if _is_shelf_zone(a) and _is_shelf_zone(b) and a != b
    )
    return min(irregular / total, 1.0)


# ---------------------------------------------------------------------------
# Public stateful manager
# ---------------------------------------------------------------------------

class BehaviorTracker:
    """
    Manages per-person state across frames and exposes feature snapshots.
    """

    def __init__(self, fps: float = 25.0) -> None:
        self._fps    = fps
        self._states: Dict[int, _PersonState] = {}

    def update(
        self,
        person: TrackedPerson,
        zone_graph: ZoneTransitionGraph,
    ) -> BehaviorFeatures:
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
        state = self._states.get(person_id)
        return state.build_features() if state else None

    def all_features(self) -> Dict[int, BehaviorFeatures]:
        return {pid: state.build_features() for pid, state in self._states.items()}

    def tracked_ids(self) -> List[int]:
        return list(self._states.keys())
