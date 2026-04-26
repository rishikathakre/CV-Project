"""
IoU-based multi-object tracker with Hungarian assignment.

Replaces DeepSORT's appearance-CNN matching with pure spatial IoU matching.
More reliable for fixed-camera, low-person-count scenes where CNN appearance
features are unreliable (e.g. old/low-resolution footage like CAVIAR 384×288).

Matching pipeline per frame:
  1. Predict each track's next bbox via smoothed velocity.
  2. Build IoU matrix (tracks × detections).
  3. Hungarian algorithm → optimal assignment above iou_threshold.
  4. Age unmatched tracks; delete tracks older than max_age frames.
  5. Spawn new tracks for unmatched detections.
"""

from typing import Dict, List, Tuple

import numpy as np

from shared.data_types import TrackedPerson


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(b1: Tuple, b2: Tuple) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _centroid_score(b1: Tuple, b2: Tuple, max_px: float = 120.0) -> float:
    """
    Proximity score based on centroid distance, capped at max_px pixels.
    Returns a value in [0, 0.3] so it never outbids a genuine IoU match
    but can bridge a gap when there is zero bbox overlap.
    Used only for tracks lost for several frames.
    """
    cx1 = b1[0] + b1[2] / 2
    cy1 = b1[1] + b1[3] / 2
    cx2 = b2[0] + b2[2] / 2
    cy2 = b2[1] + b2[3] / 2
    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    if dist >= max_px:
        return 0.0
    return 0.3 * (1.0 - dist / max_px)


# ---------------------------------------------------------------------------
# Hungarian assignment (pure numpy — no scipy dependency)
# ---------------------------------------------------------------------------

def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    Minimal Hungarian algorithm on a cost matrix.
    Returns list of (row, col) pairs for optimal assignment.
    Works on non-square matrices by padding.
    """
    n, m = cost.shape
    size = max(n, m)
    padded = np.full((size, size), cost.max() + 1.0)
    padded[:n, :m] = cost

    # Standard augmenting-path Hungarian implementation.
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minVal = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = padded[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minVal[j]:
                        minVal[j] = cur
                        way[j] = j0
                    if minVal[j] < delta:
                        delta = minVal[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minVal[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    pairs = []
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] - 1 < n and j - 1 < m:
            pairs.append((p[j] - 1, j - 1))
    return pairs


# ---------------------------------------------------------------------------
# Single track state
# ---------------------------------------------------------------------------

class _Track:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int]) -> None:
        self.track_id = track_id
        self.bbox     = bbox
        self.age      = 0    # frames since last matched
        self.hits     = 1
        self._vx      = 0.0
        self._vy      = 0.0

    _VELOCITY_CUTOFF = 5  # stop extrapolating after this many missed frames

    def predict(self) -> Tuple[int, int, int, int]:
        # Beyond a few missed frames, velocity prediction becomes unreliable.
        # Return the last known position so IoU matching stays grounded.
        if self.age >= self._VELOCITY_CUTOFF:
            return self.bbox
        x, y, w, h = self.bbox
        return (int(x + self._vx), int(y + self._vy), w, h)

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        x, y, w, h = bbox
        px, py, _, _ = self.bbox
        # Exponential moving average for velocity (α = 0.5)
        self._vx = 0.5 * self._vx + 0.5 * (x - px)
        self._vy = 0.5 * self._vy + 0.5 * (y - py)
        self.bbox = bbox
        self.age  = 0
        self.hits += 1


# ---------------------------------------------------------------------------
# Public tracker
# ---------------------------------------------------------------------------

class PersonTracker:
    """
    IoU-based multi-object tracker.

    Args:
        max_age:       Frames to keep a lost track alive before deleting it.
                       90 @ 25 fps ≈ 3.6 s.  Increase if people keep splitting.
        iou_threshold: Minimum IoU for a detection to be matched to a track.
                       Lower values allow matching across larger displacements.
    """

    def __init__(self, max_age: int = 150, iou_threshold: float = 0.10) -> None:
        self._tracks:       Dict[int, _Track] = {}
        self._next_id:      int = 1
        self._max_age:      int = max_age
        self._iou_threshold: float = iou_threshold

    def update(
        self,
        detections: List[tuple],
        frame: np.ndarray,
        frame_number: int,
        fps: float,
    ) -> List[TrackedPerson]:
        """
        Update tracker with detections from one frame.

        Args:
            detections:   output of PersonDetector.detect() —
                          list of ((x,y,w,h), confidence, label).
            frame:        raw BGR frame (unused — kept for API compatibility).
            frame_number: current frame index (0-based).
            fps:          video frame rate for timestamp computation.

        Returns:
            List of TrackedPerson for all active tracks.
        """
        det_bboxes = [d[0] for d in detections]
        track_ids  = list(self._tracks.keys())

        # ---- Match detections to tracks via IoU ----
        matched_track_to_det: Dict[int, int] = {}
        unmatched_dets = set(range(len(det_bboxes)))

        if track_ids and det_bboxes:
            # Build score matrix: rows = tracks, cols = detections.
            # Primary score is IoU.  For tracks that have been lost for several
            # frames (age > _VELOCITY_CUTOFF), also consider centroid proximity
            # so a person re-entering near their last known position still matches
            # even when bboxes don't overlap.
            score_mat = np.zeros((len(track_ids), len(det_bboxes)))
            for i, tid in enumerate(track_ids):
                pred = self._tracks[tid].predict()
                age  = self._tracks[tid].age
                for j, db in enumerate(det_bboxes):
                    iou_val = _iou(pred, db)
                    if iou_val > 0:
                        score_mat[i, j] = iou_val
                    elif age >= _Track._VELOCITY_CUTOFF:
                        # Centroid fallback for long-lost tracks (capped at 0.3)
                        score_mat[i, j] = _centroid_score(pred, db)

            for r, c in _hungarian(-score_mat):
                if score_mat[r, c] >= self._iou_threshold:
                    matched_track_to_det[track_ids[r]] = c
                    unmatched_dets.discard(c)

        # ---- Update matched tracks ----
        for tid, c in matched_track_to_det.items():
            self._tracks[tid].update(det_bboxes[c])

        # ---- Age unmatched tracks; remove stale ones ----
        for tid in track_ids:
            if tid not in matched_track_to_det:
                self._tracks[tid].age += 1
        for tid in [t for t, tr in self._tracks.items() if tr.age > self._max_age]:
            del self._tracks[tid]

        # ---- Spawn new tracks for unmatched detections ----
        for j in sorted(unmatched_dets):
            self._tracks[self._next_id] = _Track(self._next_id, det_bboxes[j])
            self._next_id += 1

        # ---- Return only recently-detected tracks ----
        # Tracks with age > _DISPLAY_AGE are kept alive internally for
        # re-identification but are hidden from the display to avoid
        # ghost boxes appearing after a person leaves the scene.
        _DISPLAY_AGE = 2
        timestamp = frame_number / fps if fps > 0 else 0.0
        return [
            TrackedPerson(
                id=t.track_id,
                frame=frame_number,
                timestamp=timestamp,
                bbox=t.bbox,
                zone="unknown",
            )
            for t in self._tracks.values()
            if t.age <= _DISPLAY_AGE
        ]
