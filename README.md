# Retail Suspicious Behaviour Detection

Real-time computer vision system that watches retail store footage and flags people who show suspicious behaviour. It uses YOLOv8 to detect people, a Kalman filter to track them across frames, and a rule-based scoring system to raise alerts.

---

## Quick start (4 steps)

### Option A - Docker (recommended, no local Python required)

**Step 1.** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and make sure it is running.

**Step 2.** Clone or unzip the project folder.

**Step 3.** Build and launch with one command:
```bash
docker compose up --build
```

**Step 4.** Open **http://localhost:8501** in your browser.

To stop: press `Ctrl+C`, then run `docker compose down`.

---

### Option B - Local Python

**Step 1.** Make sure Python 3.10 or newer is installed.

**Step 2.** Clone or unzip the project folder.

**Step 3.** Install dependencies and start the dashboard:
```bash
bash run.sh
```

Or manually:
```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

**Step 4.** Open **http://localhost:8501** in your browser.

---

## How to use the dashboard

1. Upload a video file (`.mpg`, `.mp4`, `.avi`) using the sidebar.
2. Choose a zone layout mode:
   - **Draw custom zones** - drag rectangles onto the first frame to mark shelves, walkway, billing, and exit.
   - **Auto** - zones are generated automatically from the frame size.
   - **CAVIAR config** - fixed zones for the 384×288 CAVIAR dataset.
3. Optionally enable **Save annotated video** to write the output to `data/output/`.
4. Press **▶ Run pipeline** to start processing.
5. Watch the live feed, suspicion scores, heatmap, alerts, and SHAP explanations update in real time.
6. After the run, upload a ground truth JSON file to see accuracy, F1, and the grid search results.

---

## Project structure

```
src/
  detection/        YOLOv8 person detector
  tracking/         Kalman filter + appearance multi-person tracker
  behavior/         Dwell time, revisit counting, trajectory analysis
  zone_graph/       Zone definitions and transition graph
  alerts/           Rule-based risk scoring and alert classification
  explainability/   SHAP explanations for each person's risk score
  evaluation/       Classification metrics and weight grid search
  dashboard/        Streamlit dashboard (main entry point)
shared/             Core dataclasses used across modules
configs/            Store zone layout YAML files
data/
  raw/              Input video files
  output/           Annotated video outputs saved by the dashboard
tests/              Unit tests, scenario simulations, and benchmarks
```

---

## Alert levels

| Level  | Meaning |
|--------|---------|
| NONE   | No suspicious behaviour detected |
| LOW    | Small score with minor indicators |
| MEDIUM | Long shelf dwell or moderately high score |
| HIGH   | Billing bypass, repeated zone revisits, or very high score |

---

## Model

The project uses **YOLOv8n** (`yolov8n.pt`) for person detection. The weight file is included in the repository root. No retraining is needed - the system runs inference only.

---

## Running the tests

```bash
python -m pytest tests/
```

To run the synthetic benchmark that evaluates 30 labelled scenarios:
```bash
python tests/synthetic_benchmark.py
```

To regenerate the confusion matrix from the ground truth videos:
```bash
python tests/plot_confusion_matrix.py
```
