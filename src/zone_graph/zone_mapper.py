"""
Zone mapper: maps a person's foot position to a named store zone.

Supports three modes:
  - ZoneMapper(config_path)      load from YAML  (CAVIAR-specific pixel coords)
  - ZoneMapper.from_frame(h, w)  auto-generate proportional zones for any video
  - zone_mapper = None           disable zone features entirely
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


# Proportional layout used by from_frame() — fractions of frame height.
# Adjust these if your camera angle is very different.
_AUTO_TOP_FRAC   = 0.22   # billing + exit band
_AUTO_SHELF_FRAC = 0.38   # shelves band
_AUTO_WALK_FRAC  = 0.22   # walkway
_AUTO_ENT_FRAC   = 0.18   # entrance (bottom)


class ZoneMapper:
    """Loads or generates zone polygons and classifies foot positions."""

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, config_path: str = "configs/store_layout.yaml"):
        """Load zones from a YAML file (original CAVIAR-specific coords)."""
        self._zones: Dict[str, np.ndarray] = self._load_zones(config_path)
        self._mode = "yaml"

    @classmethod
    def from_dict(cls, zones: Dict[str, np.ndarray]) -> "ZoneMapper":
        """Create a ZoneMapper from a pre-built {name: polygon_array} dict."""
        obj = cls.__new__(cls)
        obj._zones = {k: np.array(v, dtype=np.float32) for k, v in zones.items()}
        obj._mode = "custom"
        return obj

    @classmethod
    def from_frame(cls, frame_h: int, frame_w: int) -> "ZoneMapper":
        """
        Auto-generate a proportional zone grid from frame dimensions.
        Works for any camera/resolution without a config file.

        Layout (top → bottom):
          billing (left ⅓)  |  <centre top>  |  exit (right ⅓)
          shelves_left       |  shelves_center |  shelves_right
                         walkway
                         entrance
        """
        obj = cls.__new__(cls)
        obj._zones = cls._auto_zones(frame_h, frame_w)
        obj._mode = "auto"
        return obj

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_zone(self, bbox: Tuple[int, int, int, int]) -> str:
        """Return the zone name for the foot position derived from bbox."""
        foot = self._foot_position(bbox)
        return self._classify(foot)

    def zone_names(self) -> List[str]:
        return list(self._zones.keys())

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

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
    def _auto_zones(h: int, w: int) -> Dict[str, np.ndarray]:
        """Divide the frame into named zones proportionally."""
        lx = w // 3          # left-third x boundary
        rx = 2 * w // 3      # right-third x boundary

        top_y   = int(h * _AUTO_TOP_FRAC)
        shelf_y = int(h * (_AUTO_TOP_FRAC + _AUTO_SHELF_FRAC))
        walk_y  = int(h * (_AUTO_TOP_FRAC + _AUTO_SHELF_FRAC + _AUTO_WALK_FRAC))

        def rect(x1, y1, x2, y2):
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        return {
            "billing":        rect(0,   0,    lx,  top_y),
            "exit":           rect(rx,  0,    w,   top_y),
            "shelves_left":   rect(0,   top_y, lx,  shelf_y),
            "shelves_center": rect(lx,  top_y, rx,  shelf_y),
            "shelves_right":  rect(rx,  top_y, w,   shelf_y),
            "walkway":        rect(0,   shelf_y, w, walk_y),
            "entrance":       rect(0,   walk_y,  w, h),
        }

    @staticmethod
    def _foot_position(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = bbox
        return (x + w / 2.0, y + float(h))

    def _classify(self, point: Tuple[float, float]) -> str:
        pt = (float(point[0]), float(point[1]))
        for name, polygon in self._zones.items():
            result = cv2.pointPolygonTest(polygon, pt, measureDist=False)
            if result >= 0:
                return name
        return "unknown"
