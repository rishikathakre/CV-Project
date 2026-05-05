from typing import Dict, List, Tuple

import cv2
import numpy as np

from shared.data_types import TrackedPerson


# --- Coordinate helpers ---

def _to_cxcywh(bbox: Tuple) -> np.ndarray:
    """Convert (x, y, w, h) top-left format to (cx, cy, w, h) center format."""
    x, y, w, h = bbox
    return np.array([x + w / 2, y + h / 2, float(w), float(h)])


def _to_xywh(state: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert center-format state back to (x, y, w, h) top-left format."""
    cx, cy, w, h = state[0], state[1], state[2], state[3]
    return (int(cx - w / 2), int(cy - h / 2), int(max(1, w)), int(max(1, h)))


# --- Appearance descriptor ---

_HIST_BINS  = 16   # histogram bins per channel (H and S)
_HIST_ALPHA = 0.5  # how quickly the stored histogram adapts to new detections


def _extract_histogram(frame: np.ndarray, bbox: Tuple) -> np.ndarray:
    """Compute a normalised HSV histogram from the inner 60% of a bounding box.

    Using only the center crop reduces the effect of background pixels
    at the edges of the box, which are often not part of the person.
    """
    x, y, w, h = bbox
    mx, my = max(1, int(w * 0.20)), max(1, int(h * 0.20))
    x1, y1 = max(0, x + mx), max(0, y + my)
    x2, y2 = min(frame.shape[1], x + w - mx), min(frame.shape[0], y + h - my)

    if x2 <= x1 or y2 <= y1 or frame.size == 0:
        return np.full(_HIST_BINS * _HIST_BINS, 1.0 / (_HIST_BINS * _HIST_BINS))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.full(_HIST_BINS * _HIST_BINS, 1.0 / (_HIST_BINS * _HIST_BINS))

    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None,
                        [_HIST_BINS, _HIST_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Return the Bhattacharyya coefficient (0–1) measuring how similar two histograms are."""
    return float(np.clip(np.sum(np.sqrt(np.clip(h1, 0, None) *
                                        np.clip(h2, 0, None))), 0.0, 1.0))


# --- IoU ---

def _iou(b1: Tuple, b2: Tuple) -> float:
    """Return the Intersection over Union of two (x, y, w, h) bounding boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# --- Hungarian algorithm (pure numpy) ---

def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Find the assignment that minimises total cost using the Hungarian algorithm.

    The cost matrix does not have to be square. We pad it to square internally.
    Returns a list of (row, col) pairs for matched elements.
    """
    n, m   = cost.shape
    size   = max(n, m)
    padded = np.full((size, size), cost.max() + 1.0)
    padded[:n, :m] = cost

    u, v   = np.zeros(size + 1), np.zeros(size + 1)
    p, way = np.zeros(size + 1, dtype=int), np.zeros(size + 1, dtype=int)

    for i in range(1, size + 1):
        p[0] = i
        j0   = 0
        minV = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = padded[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minV[j]:
                        minV[j] = cur
                        way[j]  = j0
                    if minV[j] < delta:
                        delta = minV[j]
                        j1    = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j]    -= delta
                else:
                    minV[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0    = way[j0]

    return [(p[j] - 1, j - 1) for j in range(1, size + 1)
            if p[j] != 0 and p[j] - 1 < n and j - 1 < m]


# --- Kalman-filtered track ---

class _KalmanTrack:
    """A SORT-style Kalman filter that tracks one person.

    State vector:       [cx, cy, w, h, vx, vy]
    Observation vector: [cx, cy, w, h]

    All noise matrices are scaled to the bounding box size when the track
    is first created, so the filter works the same way for small videos
    (like CAVIAR at 384x288) and large HD footage.
    """

    # Constant-velocity model: add velocity to position each frame.
    _F = np.array([[1, 0, 0, 0, 1, 0],
                   [0, 1, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]], dtype=float)

    # Measurement matrix: maps the full state to what we can observe.
    _H = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0]], dtype=float)

    def __init__(self, track_id: int, bbox: Tuple, hist: np.ndarray) -> None:
        self.track_id = track_id
        self.hist     = hist.copy()
        self.hits     = 1   # number of frames where a detection was matched
        self.age      = 0   # frames since the last matched detection

        s = _to_cxcywh(bbox)
        w, h = s[2], s[3]

        self.x = np.array([s[0], s[1], w, h, 0.0, 0.0])

        # Scale covariances to the bbox size so uncertainty is relative to person size.
        self.P = np.diag([w * 2, h * 2, w * 4, h * 4, w * 10, h * 10])
        self.Q = np.diag([w * 0.05, h * 0.05,
                          w * 0.5,  h * 0.5,
                          w * 0.25, h * 0.25])
        self.R = np.diag([w * 0.5, h * 0.5, w * 1.0, h * 1.0])

    def predict(self) -> Tuple[int, int, int, int]:
        """Advance the state by one frame using the motion model."""
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q
        # Keep width and height positive to avoid numerical drift.
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return _to_xywh(self.x)

    def update(self, bbox: Tuple, hist: np.ndarray) -> None:
        """Correct the state using a new matched detection."""
        z = _to_cxcywh(bbox)
        y = z - self._H @ self.x           # innovation: difference between observed and predicted
        S = self._H @ self.P @ self._H.T + self.R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self._H) @ self.P
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        self.hits += 1
        self.age   = 0
        # Slowly update the stored histogram so appearance adapts to lighting changes.
        self.hist  = _HIST_ALPHA * hist + (1.0 - _HIST_ALPHA) * self.hist

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return _to_xywh(self.x)


# --- Score matrix and matching ---

_IOU_W  = 0.60  # weight given to spatial overlap when matching
_HIST_W = 0.40  # weight given to appearance similarity when matching


def _build_scores(track_ids, tracks, det_bboxes, det_hists) -> np.ndarray:
    """Build a score matrix: rows = tracks, cols = detections, values = match quality."""
    mat = np.zeros((len(track_ids), len(det_bboxes)))
    for i, tid in enumerate(track_ids):
        pred = _to_xywh(tracks[tid].x)
        for j, db in enumerate(det_bboxes):
            iou_v  = _iou(pred, db)
            hist_v = _bhattacharyya(tracks[tid].hist, det_hists[j])
            # Only assign a score when there is actual spatial overlap.
            if iou_v > 0:
                mat[i, j] = _IOU_W * iou_v + _HIST_W * hist_v
    return mat


def _run_hungarian(track_ids, score_mat, threshold) -> Tuple[dict, List[int]]:
    """Run the Hungarian algorithm on a score matrix and return matched pairs.

    Any pair where the combined score is below threshold is treated as unmatched.
    """
    matched, unmatched_tracks = {}, []
    if score_mat.size == 0:
        return matched, list(range(len(track_ids)))
    for r, c in _hungarian(-score_mat):  # negate because _hungarian minimises
        if score_mat[r, c] >= threshold:
            matched[track_ids[r]] = c
        else:
            unmatched_tracks.append(track_ids[r])
    unmatched_tracks += [track_ids[r] for r in range(len(track_ids))
                         if track_ids[r] not in matched]
    return matched, unmatched_tracks


# --- Public tracker ---

class PersonTracker:
    """Multi-person tracker using a Kalman filter for motion and histograms for appearance.

    Tracks are split into 'confirmed' (seen in at least n_init frames) and
    'tentative' (newly created, not yet confirmed). Confirmed tracks are matched
    first with a strict threshold. Leftover detections are then tried against
    tentative tracks with a relaxed threshold.
    """

    # Only show a track if it was matched within the last N frames.
    _DISPLAY_AGE = 4

    def __init__(
        self,
        max_age:       int   = 30,
        iou_threshold: float = 0.25,
        n_init:        int   = 2,
    ) -> None:
        self._tracks:         Dict[int, _KalmanTrack] = {}
        self._next_id:        int   = 1
        self._max_age:        int   = max_age
        self._iou_threshold:  float = iou_threshold
        self._n_init:         int   = n_init

    def update(
        self,
        detections:   List[tuple],
        frame:        np.ndarray,
        frame_number: int,
        fps:          float,
    ) -> List[TrackedPerson]:
        """Process one frame of detections and return the list of currently active tracks."""
        det_bboxes = [d[0] for d in detections]
        det_hists  = [_extract_histogram(frame, b) for b in det_bboxes]

        # Predict where each track will be in this frame.
        for tr in self._tracks.values():
            tr.predict()

        confirmed = [tid for tid, t in self._tracks.items() if t.hits >= self._n_init]
        tentative = [tid for tid, t in self._tracks.items() if t.hits <  self._n_init]

        matched_all:    Dict[int, int] = {}
        unmatched_dets: set            = set(range(len(det_bboxes)))

        # Stage 1: match confirmed tracks with a strict score threshold.
        if confirmed and det_bboxes:
            mat1 = _build_scores(confirmed, self._tracks, det_bboxes, det_hists)
            m1, _ = _run_hungarian(confirmed, mat1, self._iou_threshold)
            matched_all.update(m1)
            unmatched_dets -= set(m1.values())

        # Stage 2: try to match tentative tracks to leftover detections with a lower bar.
        rem_dets   = sorted(unmatched_dets)
        rem_bboxes = [det_bboxes[j] for j in rem_dets]
        rem_hists  = [det_hists[j]  for j in rem_dets]

        if tentative and rem_bboxes:
            mat2 = _build_scores(tentative, self._tracks, rem_bboxes, rem_hists)
            m2, _ = _run_hungarian(tentative, mat2, self._iou_threshold * 0.6)
            for tid, local_j in m2.items():
                global_j = rem_dets[local_j]
                matched_all[tid] = global_j
                unmatched_dets.discard(global_j)

        # Update Kalman state for every matched track.
        for tid, j in matched_all.items():
            self._tracks[tid].update(det_bboxes[j], det_hists[j])

        # Age out tracks that did not get a match this frame.
        for tid in list(self._tracks):
            if tid not in matched_all:
                self._tracks[tid].age += 1

        # Delete tracks that have been unmatched for too long.
        for tid in [t for t, tr in self._tracks.items() if tr.age > self._max_age]:
            del self._tracks[tid]

        # Create a new tentative track for each unmatched detection.
        for j in sorted(unmatched_dets):
            self._tracks[self._next_id] = _KalmanTrack(
                self._next_id, det_bboxes[j], det_hists[j]
            )
            self._next_id += 1

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
            if t.hits >= self._n_init and t.age <= self._DISPLAY_AGE
        ]
