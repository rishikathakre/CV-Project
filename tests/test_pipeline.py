"""
Integration smoke-test for the detection → tracking → zone-mapping pipeline.

Usage (from project root):
    python tests/test_pipeline.py [path/to/video.mpg]

Defaults to the first .mpg found in data/raw/ when no argument is given.
Runs on the first MAX_FRAMES frames and prints TrackedPerson output.
"""

import sys
from pathlib import Path

# Allow imports from project root regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.zone_graph.zone_mapper import ZoneMapper

MAX_FRAMES = 100


def find_default_video() -> Path:
    candidates = sorted(Path(PROJECT_ROOT / "data" / "raw").glob("*.mpg"))
    if not candidates:
        raise FileNotFoundError("No .mpg files found in data/raw/")
    return candidates[0]


def run(video_path: Path) -> None:
    print(f"Video : {video_path.name}")
    print(f"Frames: first {MAX_FRAMES}\n")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    detector = PersonDetector()
    tracker = PersonTracker()
    zone_mapper = ZoneMapper(str(PROJECT_ROOT / "configs" / "store_layout.yaml"))

    frame_idx = 0
    total_persons = 0

    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print(f"[end of video at frame {frame_idx}]")
            break

        detections = detector.detect(frame)
        tracked_persons = tracker.update(detections, frame, frame_idx, fps)

        # Attach zone to each person.
        for person in tracked_persons:
            person.zone = zone_mapper.get_zone(person.bbox)
            total_persons += 1
            print(
                f"frame={person.frame:4d}  "
                f"t={person.timestamp:6.2f}s  "
                f"id={person.id:3d}  "
                f"bbox={person.bbox}  "
                f"zone={person.zone}"
            )

        if not tracked_persons and frame_idx % 10 == 0:
            print(f"frame={frame_idx:4d}  — no confirmed tracks")

        frame_idx += 1

    cap.release()
    print(f"\nDone. Processed {frame_idx} frames, logged {total_persons} person-frame records.")


if __name__ == "__main__":
    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_default_video()
    run(video_path)
