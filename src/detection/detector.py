"""
YOLOv8 person detector.
Loads YOLOv8n and returns raw detections as (bbox, confidence, label) tuples.
Only class 0 (person) detections above the confidence threshold are returned.
"""

from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

# (x, y, w, h), confidence, label
Detection = Tuple[Tuple[int, int, int, int], float, str]

_CONFIDENCE_THRESHOLD = 0.4
_PERSON_CLASS_ID = 0


class PersonDetector:
    """Wraps YOLOv8 to detect people in a single BGR frame."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        # Model downloads automatically on first instantiation.
        self._model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on one BGR frame.

        Returns a list of (bbox, confidence, label) where bbox is (x, y, w, h)
        in pixel coordinates (top-left origin).
        """
        results = self._model(
            frame,
            classes=[_PERSON_CLASS_ID],
            conf=_CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id != _PERSON_CLASS_ID:
                    continue
                conf = float(box.conf[0])
                # xyxy → xywh
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                detections.append((bbox, conf, "person"))

        return detections
