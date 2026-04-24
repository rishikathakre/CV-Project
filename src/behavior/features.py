"""
Per-person behavioral feature extraction.

Maintains a running state dictionary keyed by tracking ID so that features
accumulate correctly across frames.  Call update() once per frame for each
TrackedPerson, then call get_features() to retrieve the current BehaviorFeatures.
"""

from typing import Dict, List, Optional

from shared.data_types import BehaviorFeatures, TrackedPerson
from src.zone_graph.graph import ZoneTransitionGraph


# Zones that must be visited before reaching exit to avoid billing_bypassed.
_BILLING_ZONE = "billing"
_EXIT_ZONE = "exit"

# Zones considered "shelf" zones for trajectory-irregularity heuristic.
_SHELF_ZONES = {"shelves_left", "shelves_center", "shelves_right"}


class _PersonState:
    """Internal mutable state for a single tracked person."""

    def __init__(self, person_id: int, fps: float) -> None:
        self.id = person_id
        self.fps = fps

        # Dwell tracking: zone → frame count in that zone.
        self.dwell_frames: Dict[str, int] = {}

        # Revisit tracking: zone → number of *re-entries* (first entry not counted).
        self.zone_revisits: Dict[str, int] = {}

        # Ordered sequence of zone transitions (consecutive duplicates collapsed).
        self.zone_sequence: List[str] = []

        # Set of zones the person has visited at all.
        self.visited_zones: set = set()

        # Whether billing was seen before any exit visit.
        self.billing_seen: bool = False
        self.billing_bypassed: bool = False

        # Last zone recorded (used to detect transitions).
        self.last_zone: Optional[str] = None

    # ------------------------------------------------------------------

    def update(self, zone: str) -> Optional[str]:
        """
        Process one frame for this person.

        Returns the previous zone name if a zone transition occurred
        (useful for the caller to record in the graph), else None.
        """
        transition_from: Optional[str] = None

        # Dwell accumulation.
        self.dwell_frames[zone] = self.dwell_frames.get(zone, 0) + 1

        # Zone transition / revisit detection.
        if zone != self.last_zone:
            transition_from = self.last_zone  # may be None on first frame

            # Revisit: person has been here before and is entering again.
            if zone in self.visited_zones:
                self.zone_revisits[zone] = self.zone_revisits.get(zone, 0) + 1
            self.visited_zones.add(zone)

            # Append to ordered sequence (collapse consecutive duplicates).
            self.zone_sequence.append(zone)
            self.last_zone = zone

        # Billing-bypass detection.
        if zone == _BILLING_ZONE:
            self.billing_seen = True
        if zone == _EXIT_ZONE and not self.billing_seen:
            self.billing_bypassed = True

        return transition_from

    def build_features(self) -> BehaviorFeatures:
        """Snapshot the current state as a BehaviorFeatures instance."""
        dwell_seconds = {
            z: frames / self.fps for z, frames in self.dwell_frames.items()
        }
        irregularity = _compute_trajectory_irregularity(self.zone_sequence)

        return BehaviorFeatures(
            id=self.id,
            dwell_per_zone=dict(dwell_seconds),
            zone_revisits=dict(self.zone_revisits),
            zone_sequence=list(self.zone_sequence),
            billing_bypassed=self.billing_bypassed,
            trajectory_irregularity=irregularity,
            suspicion_score=0.0,      # filled in by scoring.py
            alert_reasons=[],         # filled in by explainer.py
        )


def _compute_trajectory_irregularity(zone_sequence: List[str]) -> float:
    """
    Heuristic: ratio of back-and-forth shelf transitions to total transitions.

    A normal shopper moves forward; repeated shelf ↔ shelf toggling is irregular.
    Returns a value in [0.0, 1.0].
    """
    if len(zone_sequence) < 2:
        return 0.0

    transitions = list(zip(zone_sequence, zone_sequence[1:]))
    total = len(transitions)
    irregular = sum(
        1
        for a, b in transitions
        if a in _SHELF_ZONES and b in _SHELF_ZONES and a != b
    )
    return min(irregular / total, 1.0)


# ---------------------------------------------------------------------------
# Public stateful manager
# ---------------------------------------------------------------------------

class BehaviorTracker:
    """
    Manages per-person state across frames and exposes feature snapshots.

    Usage:
        tracker = BehaviorTracker(fps=25.0)
        for frame_persons in pipeline:
            for person in frame_persons:
                features = tracker.update(person, zone_graph)
    """

    def __init__(self, fps: float = 25.0) -> None:
        self._fps = fps
        self._states: Dict[int, _PersonState] = {}

    def update(
        self,
        person: TrackedPerson,
        zone_graph: ZoneTransitionGraph,
    ) -> BehaviorFeatures:
        """
        Process one TrackedPerson observation.

        Updates internal state, records the transition in zone_graph if the
        person changed zones, and returns the latest BehaviorFeatures snapshot.
        """
        pid = person.id
        zone = person.zone

        if pid not in self._states:
            self._states[pid] = _PersonState(pid, self._fps)

        state = self._states[pid]
        prev_zone = state.update(zone)

        # Record transition in the shared graph when zone actually changed.
        if prev_zone is not None:
            zone_graph.add_transition(prev_zone, zone)

        return state.build_features()

    def get_features(self, person_id: int) -> Optional[BehaviorFeatures]:
        """Return the latest BehaviorFeatures for a known person, or None."""
        state = self._states.get(person_id)
        return state.build_features() if state else None

    def all_features(self) -> Dict[int, BehaviorFeatures]:
        """Return current BehaviorFeatures for every tracked person."""
        return {pid: state.build_features() for pid, state in self._states.items()}

    def tracked_ids(self) -> List[int]:
        return list(self._states.keys())
