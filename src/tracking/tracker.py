"""
DeepSORT multi-object tracker.
Wraps deep-sort-realtime to maintain persistent person IDs across frames
and returns a list of TrackedPerson dataclass instances.
Only confirmed tracks are returned to avoid noise from tentative detections.
"""

from typing import List

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from shared.data_types import TrackedPerson


class PersonTracker:
    """Assigns stable IDs to detected people across video frames."""

    def __init__(self, max_age: int = 30, n_init: int = 3):
        # max_age: frames to keep a lost track alive before deletion.
        # n_init:  consecutive detections required before a track is confirmed.
        self._tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(
        self,
        detections: List[tuple],
        frame: np.ndarray,
        frame_number: int,
        fps: float,
    ) -> List[TrackedPerson]:
        """
        Update tracker with detections from the current frame.

        Args:
            detections: output of PersonDetector.detect() —
                        list of ((x,y,w,h), confidence, label) tuples.
            frame:       raw BGR frame (used by DeepSORT for appearance embedding).
            frame_number: current frame index (0-based).
            fps:          video frame rate, used to compute timestamp.

        Returns:
            List of TrackedPerson objects for all confirmed tracks this frame.
        """
        # DeepSORT expects detections as ([x,y,w,h], confidence, label).
        ds_input = [
            ([bbox[0], bbox[1], bbox[2], bbox[3]], conf, label)
            for bbox, conf, label in detections
        ]

        tracks = self._tracker.update_tracks(ds_input, frame=frame)

        timestamp = frame_number / fps if fps > 0 else 0.0
        tracked: List[TrackedPerson] = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()           # left, top, right, bottom
            x1, y1, x2, y2 = (int(v) for v in ltrb)
            bbox = (x1, y1, x2 - x1, y2 - y1)

            tracked.append(
                TrackedPerson(
                    id=int(track.track_id),
                    frame=frame_number,
                    timestamp=timestamp,
                    bbox=bbox,
                    zone="unknown",           # filled in by zone_mapper
                )
            )

        return tracked
