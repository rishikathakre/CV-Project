"""
SORT-style Kalman filter tracker with HSV appearance matching.

Each track maintains a 6-D Kalman state [cx, cy, w, h, vx, vy].
The Kalman predict step gives a more accurate expected position than
simple velocity extrapolation, which keeps the IoU matrix sharp even
in crowded scenes where bounding boxes heavily overlap.

Matching pipeline per frame:
  1. Kalman predict — each track estimates its new position.
  2. Extract HSV colour histograms for every new detection.
  3. Build a combined score matrix (IoU × appearance).
  4. Two-stage Hungarian assignment:
       Stage 1 — confirmed tracks (hits ≥ n_init) vs all detections,
                 strict threshold.
       Stage 2 — unconfirmed + still-unmatched tracks vs remaining
                 detections, relaxed threshold.
  5. Age unmatched tracks; delete tracks older than max_age frames.
  6. Spawn new tracks for unmatched detections.
  7. Return only confirmed, recently-matched tracks.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np

from shared.data_types import TrackedPerson


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _to_cxcywh(bbox: Tuple) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x + w / 2, y + h / 2, float(w), float(h)])


def _to_xywh(state: np.ndarray) -> Tuple[int, int, int, int]:
    cx, cy, w, h = state[0], state[1], state[2], state[3]
    return (int(cx - w / 2), int(cy - h / 2), int(max(1, w)), int(max(1, h)))


# ---------------------------------------------------------------------------
# Appearance descriptor
# ---------------------------------------------------------------------------

_HIST_BINS  = 16
_HIST_ALPHA = 0.5   # EMA weight for histogram update


def _extract_histogram(frame: np.ndarray, bbox: Tuple) -> np.ndarray:
    """Normalised HSV histogram of the inner 60 % of a bbox crop."""
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
    return float(np.clip(np.sum(np.sqrt(np.clip(h1, 0, None) *
                                        np.clip(h2, 0, None))), 0.0, 1.0))


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def _iou(b1: Tuple, b2: Tuple) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Hungarian (pure numpy)
# ---------------------------------------------------------------------------

def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
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


# ---------------------------------------------------------------------------
# Kalman-filtered track
# ---------------------------------------------------------------------------

class _KalmanTrack:
    """
    SORT-style Kalman filter for one person.
    State:       [cx, cy, w, h, vx, vy]
    Observation: [cx, cy, w, h]

    Noise matrices are scaled to the initial bbox size so the filter
    behaves the same for small (CAVIAR) and large (HD) footage.
    """

    # Constant-velocity transition  (add dt*velocity to position)
    _F = np.array([[1, 0, 0, 0, 1, 0],
                   [0, 1, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]], dtype=float)

    # Measurement maps observation onto state
    _H = np.array([[1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0]], dtype=float)

    def __init__(self, track_id: int, bbox: Tuple,
                 hist: np.ndarray) -> None:
        self.track_id = track_id
        self.hist     = hist.copy()
        self.hits     = 1
        self.age      = 0   # frames since last matched

        s = _to_cxcywh(bbox)
        w, h = s[2], s[3]

        self.x = np.array([s[0], s[1], w, h, 0.0, 0.0])

        # Covariances scaled to bbox size for resolution-independence
        self.P = np.diag([w * 2, h * 2, w * 4, h * 4, w * 10, h * 10])

        # Process noise: how much the motion model can drift per frame
        self.Q = np.diag([w * 0.05, h * 0.05,
                          w * 0.5,  h * 0.5,
                          w * 0.25, h * 0.25])

        # Measurement noise: detection uncertainty
        self.R = np.diag([w * 0.5, h * 0.5, w * 1.0, h * 1.0])

    # ------------------------------------------------------------------ #
    # Kalman steps
    # ------------------------------------------------------------------ #

    def predict(self) -> Tuple[int, int, int, int]:
        """Predict step — advance state, inflate uncertainty."""
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q
        # Clamp w/h to stay positive after numerical drift
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return _to_xywh(self.x)

    def update(self, bbox: Tuple, hist: np.ndarray) -> None:
        """Update step — fuse detection with prediction."""
        z = _to_cxcywh(bbox)
        y = z - self._H @ self.x                       # innovation
        S = self._H @ self.P @ self._H.T + self.R      # innovation covariance
        K = self.P @ self._H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self._H) @ self.P
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        self.hits += 1
        self.age   = 0
        # Smooth appearance: adapt slowly to lighting changes
        self.hist  = _HIST_ALPHA * hist + (1.0 - _HIST_ALPHA) * self.hist

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return _to_xywh(self.x)


# ---------------------------------------------------------------------------
# Score matrix + matching helper
# ---------------------------------------------------------------------------

_IOU_W  = 0.60
_HIST_W = 0.40


def _build_scores(track_ids, tracks, det_bboxes, det_hists) -> np.ndarray:
    mat = np.zeros((len(track_ids), len(det_bboxes)))
    for i, tid in enumerate(track_ids):
        pred = _to_xywh(tracks[tid].x)   # already predicted this frame
        for j, db in enumerate(det_bboxes):
            iou_v  = _iou(pred, db)
            hist_v = _bhattacharyya(tracks[tid].hist, det_hists[j])
            if iou_v > 0:
                mat[i, j] = _IOU_W * iou_v + _HIST_W * hist_v
    return mat


def _run_hungarian(track_ids, score_mat, threshold) -> Tuple[dict, List[int]]:
    matched, unmatched_tracks = {}, []
    if score_mat.size == 0:
        return matched, list(range(len(track_ids)))
    for r, c in _hungarian(-score_mat):
        if score_mat[r, c] >= threshold:
            matched[track_ids[r]] = c
        else:
            unmatched_tracks.append(track_ids[r])
    unmatched_tracks += [track_ids[r] for r in range(len(track_ids))
                         if track_ids[r] not in matched]
    return matched, unmatched_tracks


# ---------------------------------------------------------------------------
# Public tracker
# ---------------------------------------------------------------------------

class PersonTracker:
    """
    SORT-style Kalman + appearance tracker.

    Args:
        max_age:       Frames a lost track survives before deletion.
        iou_threshold: Minimum combined score (IoU+appearance) for a
                       confirmed track to match a detection.
        n_init:        Consecutive hits before a new track is displayed.
                       Suppresses one-frame ghost detections.
    """

    _DISPLAY_AGE = 4   # hide if not matched in the last N frames

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
        det_bboxes = [d[0] for d in detections]
        det_hists  = [_extract_histogram(frame, b) for b in det_bboxes]

        # ---- Kalman predict for every track ----
        for tr in self._tracks.values():
            tr.predict()

        # ---- Split confirmed vs tentative tracks ----
        confirmed   = [tid for tid, t in self._tracks.items()
                       if t.hits >= self._n_init]
        tentative   = [tid for tid, t in self._tracks.items()
                       if t.hits <  self._n_init]

        # ---- Stage 1: match confirmed tracks (strict threshold) ----
        matched_all:    Dict[int, int] = {}
        unmatched_dets: set            = set(range(len(det_bboxes)))

        if confirmed and det_bboxes:
            mat1 = _build_scores(confirmed, self._tracks, det_bboxes, det_hists)
            m1, _ = _run_hungarian(confirmed, mat1, self._iou_threshold)
            matched_all.update(m1)
            unmatched_dets -= set(m1.values())

        # ---- Stage 2: match tentative tracks vs remaining detections ----
        rem_dets    = sorted(unmatched_dets)
        rem_bboxes  = [det_bboxes[j] for j in rem_dets]
        rem_hists   = [det_hists[j]  for j in rem_dets]

        if tentative and rem_bboxes:
            mat2 = _build_scores(tentative, self._tracks, rem_bboxes, rem_hists)
            # Relaxed threshold for tentative tracks (they haven't been confirmed yet)
            m2, _ = _run_hungarian(tentative, mat2, self._iou_threshold * 0.6)
            for tid, local_j in m2.items():
                global_j = rem_dets[local_j]
                matched_all[tid] = global_j
                unmatched_dets.discard(global_j)

        # ---- Update matched tracks ----
        for tid, j in matched_all.items():
            self._tracks[tid].update(det_bboxes[j], det_hists[j])

        # ---- Age unmatched tracks ----
        for tid in list(self._tracks):
            if tid not in matched_all:
                self._tracks[tid].age += 1

        # ---- Delete stale tracks ----
        for tid in [t for t, tr in self._tracks.items()
                    if tr.age > self._max_age]:
            del self._tracks[tid]

        # ---- Spawn new tracks ----
        for j in sorted(unmatched_dets):
            self._tracks[self._next_id] = _KalmanTrack(
                self._next_id, det_bboxes[j], det_hists[j]
            )
            self._next_id += 1

        # ---- Return confirmed + recently matched tracks only ----
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
