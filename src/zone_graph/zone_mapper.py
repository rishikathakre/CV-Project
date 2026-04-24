"""
Zone mapper: maps a person's foot position (bottom-centre of their bounding box)
to a named store zone using polygon containment tests.
Zone polygons are loaded exclusively from configs/store_layout.yaml.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


class ZoneMapper:
    """Loads zone polygons and classifies (x, y) foot positions into zone names."""

    def __init__(self, config_path: str = "configs/store_layout.yaml"):
        self._zones: Dict[str, np.ndarray] = self._load_zones(config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_zone(self, bbox: Tuple[int, int, int, int]) -> str:
        """
        Return the zone name for the foot position derived from bbox.

        bbox is (x, y, w, h) with top-left origin.
        Foot position is the bottom-centre of the bounding box.
        Returns "unknown" when no zone polygon contains the foot point.
        """
        foot = self._foot_position(bbox)
        return self._classify(foot)

    def zone_names(self) -> List[str]:
        """Return all configured zone names."""
        return list(self._zones.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_zones(config_path: str) -> Dict[str, np.ndarray]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Zone config not found: {path.resolve()}")

        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh)

        zones: Dict[str, np.ndarray] = {}
        for name, data in cfg.get("zones", {}).items():
            pts = np.array(data["polygon"], dtype=np.float32)
            zones[name] = pts

        return zones

    @staticmethod
    def _foot_position(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = bbox
        return (x + w / 2.0, y + float(h))

    def _classify(self, point: Tuple[float, float]) -> str:
        pt = (float(point[0]), float(point[1]))
        for name, polygon in self._zones.items():
            # measureDist < 0 → inside, == 0 → on edge, > 0 → outside
            result = cv2.pointPolygonTest(polygon, pt, measureDist=False)
            if result >= 0:
                return name
        return "unknown"
