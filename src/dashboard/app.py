"""
Streamlit real-time retail risk dashboard.

Upload a .mpg (or any OpenCV-readable) video via the sidebar.
The full pipeline runs frame-by-frame and displays:
  - Live annotated video feed with bounding boxes and zone labels
  - Per-person suspicion score table
  - Active alerts panel
  - Cumulative zone heatmap (foot-position density)
  - Risk score timeline chart per person
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Allow imports from project root when launched as `streamlit run src/dashboard/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts.explainer import generate_alert, get_alert_level
from src.behavior.features import BehaviorTracker
from src.behavior.scoring import compute_score
from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.zone_graph.graph import ZoneTransitionGraph
from src.zone_graph.zone_mapper import ZoneMapper

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Risk Monitor",
    page_icon="🛒",
    layout="wide",
)

CONFIG_PATH = str(PROJECT_ROOT / "configs" / "store_layout.yaml")

# ---------------------------------------------------------------------------
# Colour palette: person id → BGR colour (for OpenCV drawing)
# ---------------------------------------------------------------------------
_PALETTE = [
    (0, 255, 0),    # green
    (0, 128, 255),  # orange
    (255, 0, 128),  # purple
    (0, 255, 255),  # yellow
    (255, 128, 0),  # blue
    (128, 0, 255),  # pink
    (0, 200, 200),  # teal
    (200, 200, 0),  # cyan
]


def _person_colour(person_id: int):
    return _PALETTE[person_id % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Alert colour helpers for Streamlit markdown
# ---------------------------------------------------------------------------
_ALERT_COLOURS = {"HIGH": "#FF4B4B", "MEDIUM": "#FFA500", "LOW": "#FFD700", "NONE": "#AAAAAA"}


def _alert_badge(level: str) -> str:
    colour = _ALERT_COLOURS.get(level, "#AAAAAA")
    return f'<span style="background:{colour};padding:2px 8px;border-radius:4px;color:black;font-weight:bold">{level}</span>'


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    tracked_persons,
    scores: dict,
) -> np.ndarray:
    """Draw bounding boxes, IDs, zone labels, and score on the frame."""
    out = frame.copy()
    for person in tracked_persons:
        x, y, w, h = person.bbox
        colour = _person_colour(person.id)
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        score = scores.get(person.id, 0.0)
        label = f"ID{person.id} [{person.zone}] {score:.2f}"
        cv2.putText(out, label, (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Heatmap accumulation
# ---------------------------------------------------------------------------

def _update_heatmap(heatmap: np.ndarray, tracked_persons) -> np.ndarray:
    for person in tracked_persons:
        x, y, w, h = person.bbox
        fx = int(x + w / 2)
        fy = int(y + h)
        if 0 <= fy < heatmap.shape[0] and 0 <= fx < heatmap.shape[1]:
            heatmap[fy, fx] += 1
    return heatmap


def _render_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Convert accumulator to a colour-mapped image."""
    if heatmap.max() == 0:
        return np.zeros((*heatmap.shape, 3), dtype=np.uint8)
    norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return coloured


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Retail Risk Monitoring Dashboard")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Upload Video")
        uploaded = st.file_uploader("Choose a .mpg / .mp4 file", type=["mpg", "mpeg", "mp4", "avi"])
        max_frames = st.slider("Max frames to process", 50, 2000, 500, step=50)
        conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, 0.4, step=0.05)
        run_btn = st.button("▶ Run pipeline", type="primary")

    if not uploaded or not run_btn:
        st.info("Upload a video file in the sidebar and press **Run pipeline** to start.")
        return

    # ---- Save upload to temp file ----
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open the video file. Ensure OpenCV can decode the format.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- Pipeline components ----
    detector = PersonDetector(model_name="yolov8n.pt")
    # Honour the sidebar confidence threshold.
    detector._model.overrides["conf"] = conf_threshold

    tracker = PersonTracker()
    zone_mapper = ZoneMapper(CONFIG_PATH)
    zone_graph = ZoneTransitionGraph()
    behavior_tracker = BehaviorTracker(fps=fps)

    # ---- Layout ----
    col_video, col_alerts = st.columns([3, 2])

    with col_video:
        st.subheader("Live Feed")
        video_placeholder = st.empty()

    with col_alerts:
        st.subheader("Active Alerts")
        alerts_placeholder = st.empty()

    st.divider()
    col_scores, col_heatmap = st.columns([2, 2])

    with col_scores:
        st.subheader("Suspicion Scores")
        scores_placeholder = st.empty()

    with col_heatmap:
        st.subheader("Foot-Position Heatmap")
        heatmap_placeholder = st.empty()

    st.divider()
    st.subheader("Risk Score Timeline")
    timeline_placeholder = st.empty()

    # ---- Processing state ----
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 288
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 384
    heatmap_acc = np.zeros((frame_h, frame_w), dtype=np.float32)

    # {person_id: [score_per_frame]}  for timeline chart
    score_history: dict = {}
    frame_history: dict = {}    # {person_id: [frame_number]}

    progress = st.progress(0, text="Processing frames…")
    status_text = st.empty()

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked_persons = tracker.update(detections, frame, frame_idx, fps)

        for person in tracked_persons:
            person.zone = zone_mapper.get_zone(person.bbox)

        # Accumulate heatmap.
        heatmap_acc = _update_heatmap(heatmap_acc, tracked_persons)

        # Feature extraction → scoring → alert generation.
        current_scores: dict = {}
        current_alerts: list = []

        for person in tracked_persons:
            features = behavior_tracker.update(person, zone_graph)
            features = compute_score(features)
            features = generate_alert(features)

            pid = person.id
            current_scores[pid] = features.suspicion_score

            # Build timeline history.
            if pid not in score_history:
                score_history[pid] = []
                frame_history[pid] = []
            score_history[pid].append(features.suspicion_score)
            frame_history[pid].append(frame_idx)

            level = get_alert_level(features)
            if level in ("MEDIUM", "HIGH"):
                current_alerts.append((pid, level, features.alert_reasons))

        # ---- Update UI every 5 frames to avoid redraws being too slow ----
        if frame_idx % 5 == 0 or frame_idx == max_frames - 1:
            # Annotated video frame.
            annotated = _annotate_frame(frame, tracked_persons, current_scores)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            # Alerts panel.
            if current_alerts:
                alert_html = ""
                for pid, level, reasons in current_alerts:
                    alert_html += f"<b>Person {pid}</b> {_alert_badge(level)}<br>"
                    for r in reasons[:4]:   # cap at 4 lines per person
                        alert_html += f"&nbsp;&nbsp;• {r}<br>"
                    alert_html += "<br>"
                alerts_placeholder.markdown(alert_html, unsafe_allow_html=True)
            else:
                alerts_placeholder.info("No active alerts.")

            # Scores table.
            if current_scores:
                df = pd.DataFrame(
                    [{"Person ID": pid, "Suspicion Score": f"{s:.3f}"}
                     for pid, s in sorted(current_scores.items())]
                )
                scores_placeholder.dataframe(df, use_container_width=True, hide_index=True)

            # Heatmap.
            heatmap_rgb = cv2.cvtColor(_render_heatmap(heatmap_acc), cv2.COLOR_BGR2RGB)
            heatmap_placeholder.image(heatmap_rgb, channels="RGB", use_container_width=True)

            # Timeline chart.
            if score_history:
                timeline_df = pd.DataFrame(
                    {
                        f"Person {pid}": pd.Series(scores, index=frames)
                        for pid, (scores, frames) in {
                            pid: (score_history[pid], frame_history[pid])
                            for pid in score_history
                        }.items()
                    }
                )
                timeline_placeholder.line_chart(timeline_df)

        # Progress bar.
        pct = int((frame_idx + 1) / max_frames * 100)
        progress.progress(min(pct, 100), text=f"Frame {frame_idx + 1} / {min(max_frames, total_video_frames)}")
        status_text.text(
            f"Tracking {len(tracked_persons)} person(s) | "
            f"Active IDs: {[p.id for p in tracked_persons]}"
        )

        frame_idx += 1

    cap.release()
    progress.progress(100, text="Done.")
    status_text.text(f"Processed {frame_idx} frames.")

    # ---- Final summary ----
    st.divider()
    st.subheader("Final Summary")
    all_features = behavior_tracker.all_features()
    if all_features:
        rows = []
        for pid, feat in sorted(all_features.items()):
            feat = compute_score(feat)
            feat = generate_alert(feat)
            rows.append({
                "Person ID": pid,
                "Zones Visited": ", ".join(dict.fromkeys(feat.zone_sequence)) or "—",
                "Total Revisits": sum(feat.zone_revisits.values()),
                "Billing Bypassed": "YES" if feat.billing_bypassed else "no",
                "Suspicion Score": f"{feat.suspicion_score:.3f}",
                "Alert Level": get_alert_level(feat),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No persons were tracked in the processed frames.")

    # ---- Final heatmap ----
    st.subheader("Final Heatmap")
    final_heatmap = cv2.cvtColor(_render_heatmap(heatmap_acc), cv2.COLOR_BGR2RGB)
    st.image(final_heatmap, channels="RGB", use_container_width=True)


if __name__ == "__main__":
    main()
