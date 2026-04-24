"""
Shared dataclasses used as the inter-module data contract.
All modules import TrackedPerson and BehaviorFeatures from here only.
Do NOT modify these structures without updating every module that uses them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TrackedPerson:
    """Represents a single person detection with a persistent tracking ID."""
    id: int                          # persistent tracking ID across frames
    frame: int                       # frame number
    timestamp: float                 # seconds elapsed
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    zone: str                        # zone name from zone_mapper


@dataclass
class BehaviorFeatures:
    """Aggregated behavioral features and suspicion score for one tracked person."""
    id: int
    dwell_per_zone: Dict[str, float] = field(default_factory=dict)   # seconds per zone
    zone_revisits: Dict[str, int] = field(default_factory=dict)      # re-entry counts
    zone_sequence: List[str] = field(default_factory=list)           # ordered transitions
    billing_bypassed: bool = False
    trajectory_irregularity: float = 0.0                             # 0.0–1.0
    suspicion_score: float = 0.0                                     # 0.0–1.0
    alert_reasons: List[str] = field(default_factory=list)
