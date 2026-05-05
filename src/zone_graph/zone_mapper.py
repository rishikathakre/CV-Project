from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Fractions of the frame height used to split zones automatically.
# From top to bottom: billing/exit row, shelves, walkway, entrance.
_AUTO_TOP_FRAC   = 0.22
_AUTO_SHELF_FRAC = 0.38
_AUTO_WALK_FRAC  = 0.22
_AUTO_ENT_FRAC   = 0.18


class ZoneMapper:
    """Maps a person's foot position to a named store zone.

    Zones can come from a YAML config file, be drawn by the user on the
    dashboard, or be generated automatically based on frame size.
    """

    def __init__(self, config_path: str = "configs/store_layout.yaml"):
        self._zones: Dict[str, np.ndarray] = self._load_zones(config_path)
        self._mode = "yaml"

    @classmethod
    def from_dict(cls, zones: Dict[str, np.ndarray]) -> "ZoneMapper":
        """Create a ZoneMapper from zones the user drew on the canvas."""
        obj = cls.__new__(cls)
        obj._zones = {k: np.array(v, dtype=np.float32) for k, v in zones.items()}
        obj._mode = "custom"
        return obj

    @classmethod
    def from_frame(cls, frame_h: int, frame_w: int) -> "ZoneMapper":
        """Create a ZoneMapper by dividing the frame into proportional zones automatically."""
        obj = cls.__new__(cls)
        obj._zones = cls._auto_zones(frame_h, frame_w)
        obj._mode = "auto"
        return obj

    def get_zone(self, bbox: Tuple[int, int, int, int]) -> str:
        """Return the zone name for the foot position of a bounding box."""
        foot = self._foot_position(bbox)
        return self._classify(foot)

    def zone_names(self) -> List[str]:
        return list(self._zones.keys())

    @property
    def mode(self) -> str:
        return self._mode

    @staticmethod
    def _load_zones(config_path: str) -> Dict[str, np.ndarray]:
        """Read zone polygons from a YAML file."""
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
        """Divide the frame into 7 zones using fixed height fractions."""
        lx = w // 3
        rx = 2 * w // 3
        top_y   = int(h * _AUTO_TOP_FRAC)
        shelf_y = int(h * (_AUTO_TOP_FRAC + _AUTO_SHELF_FRAC))
        walk_y  = int(h * (_AUTO_TOP_FRAC + _AUTO_SHELF_FRAC + _AUTO_WALK_FRAC))

        def rect(x1, y1, x2, y2):
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        return {
            "billing":        rect(0,   0,      lx, top_y),
            "exit":           rect(rx,  0,      w,  top_y),
            "shelves_left":   rect(0,   top_y,  lx, shelf_y),
            "shelves_center": rect(lx,  top_y,  rx, shelf_y),
            "shelves_right":  rect(rx,  top_y,  w,  shelf_y),
            "walkway":        rect(0,   shelf_y, w, walk_y),
            "entrance":       rect(0,   walk_y,  w, h),
        }

    @staticmethod
    def _foot_position(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Return the bottom-center point of a bounding box, which is where the feet are."""
        x, y, w, h = bbox
        return (x + w / 2.0, y + float(h))

    def _classify(self, point: Tuple[float, float]) -> str:
        """Check which polygon contains the point and return that zone's name."""
        pt = (float(point[0]), float(point[1]))
        for name, polygon in self._zones.items():
            result = cv2.pointPolygonTest(polygon, pt, measureDist=False)
            if result >= 0:
                return name
        return "unknown"
