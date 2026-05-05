from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

# A detection is a bounding box, a confidence value, and a class label.
Detection = Tuple[Tuple[int, int, int, int], float, str]

# Only keep detections where the model is at least this confident.
_CONFIDENCE_THRESHOLD = 0.4

# YOLO class index 0 is "person".
_PERSON_CLASS_ID = 0


class PersonDetector:
    """Runs YOLOv8 on a video frame and returns bounding boxes for all detected people."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        self._model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on one frame and return a list of (bbox, confidence, label) tuples."""
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
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                # Convert from corner format to (x, y, width, height).
                bbox = (x1, y1, x2 - x1, y2 - y1)
                detections.append((bbox, conf, "person"))
        return detections
