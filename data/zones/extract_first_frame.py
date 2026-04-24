"""
Utility: extract the first frame from a CAVIAR video and save it as first_frame.png.
Use this image as a reference when drawing zone polygons in configs/store_layout.yaml.

Usage (from project root):
    python data/zones/extract_first_frame.py [path/to/video.mpg]
"""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = Path(__file__).parent / "first_frame.png"


def main() -> None:
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        candidates = sorted(RAW_DIR.glob("*.mpg"))
        if not candidates:
            print("No .mpg files found in data/raw/")
            sys.exit(1)
        video_path = candidates[0]

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Failed to read from {video_path}")
        sys.exit(1)

    cv2.imwrite(str(OUT_PATH), frame)
    h, w = frame.shape[:2]
    print(f"Saved {OUT_PATH}  ({w}x{h} pixels)")


if __name__ == "__main__":
    main()
