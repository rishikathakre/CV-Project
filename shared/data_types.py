from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TrackedPerson:
    """One person detection result for a single video frame."""
    id: int                          # unique ID assigned by the tracker
    frame: int                       # frame number this was seen in
    timestamp: float                 # time in seconds from the start of the video
    bbox: Tuple[int, int, int, int]  # bounding box as (x, y, width, height)
    zone: str                        # store zone the person is currently in


@dataclass
class BehaviorFeatures:
    """All behavioral signals collected for one tracked person over time."""
    id: int
    dwell_per_zone: Dict[str, float] = field(default_factory=dict)   # seconds spent in each zone
    zone_revisits: Dict[str, int]    = field(default_factory=dict)   # how many times each zone was re-entered
    zone_sequence: List[str]         = field(default_factory=list)   # ordered list of zones visited
    billing_bypassed: bool = False        # True if person left without going to billing
    trajectory_irregularity: float = 0.0 # 0 to 1, higher means more erratic movement
    suspicion_score: float = 0.0          # overall risk score from 0 to 1
    alert_reasons: List[str] = field(default_factory=list)  # human-readable reasons for the alert
