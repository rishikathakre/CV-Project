# CV Project — Retail Suspicious Behavior Detection

A real-time computer vision pipeline that detects suspicious behavior in retail environments using YOLOv8 person detection, IoU-based tracking, zone transition analysis, and adaptive risk scoring. The results are presented through an interactive Streamlit dashboard.

## Running with Docker (single command)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Build and run

```bash
docker compose up --build
```

Open your browser at **http://localhost:8501**.

To stop: `Ctrl+C`, then `docker compose down`.

### What the container includes
- Python 3.11 runtime with all dependencies pre-installed
- YOLOv8n model weights (`yolov8n.pt`)
- Full application source (`src/`, `shared/`, `configs/`)
- Your local `data/` folder is mounted into the container, so uploaded videos and outputs are preserved between runs

---

## Running locally (without Docker)

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

---

## Project structure

```
src/
  detection/   # YOLOv8 person detector
  tracking/    # IoU-based multi-person tracker
  behavior/    # Dwell time, revisit, trajectory feature extraction
  zone_graph/  # Zone definitions and transition graph
  alerts/      # Risk scoring and alert classification
  dashboard/   # Streamlit UI (main entry point)
shared/        # Core dataclasses
configs/       # Store zone layout (YAML)
data/
  raw/         # Input video files
  output/      # Annotated video outputs
```

## Usage

1. Launch the app (Docker or local).
2. Upload a video file or select one from `data/raw/`.
3. Draw store zones on the first frame using the canvas tool.
4. Click **Run** to start analysis.
5. View live annotated video, suspicion scores, heatmap, and alerts in the dashboard.
