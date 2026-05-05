import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from PIL import Image as _PILImage
    import base64 as _b64, io as _io, streamlit.elements.image as _st_img
    if not hasattr(_st_img, "image_to_url"):
        def _image_to_url(image, width, clamp, channels, output_format, image_id,
                          allow_emoji=False):
            buf = _io.BytesIO()
            img = image if hasattr(image, "save") else _PILImage.fromarray(image)
            img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=80)
            b64 = _b64.b64encode(buf.getvalue()).decode()
            return f"data:image/jpeg;base64,{b64}"
        _st_img.image_to_url = _image_to_url
    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except ImportError:
    _HAS_CANVAS = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts.explainer import generate_alert, get_alert_level
from src.behavior.features import BehaviorTracker
from src.behavior.scoring import AdaptiveScorer, compute_score
from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.zone_graph.graph import ZoneTransitionGraph
from src.zone_graph.zone_mapper import ZoneMapper

from src.explainability.shap_explainer import SHAPExplainer, plot_waterfall, plot_summary_bar
from src.evaluation.metrics import classification_report
from src.evaluation.grid_search import run_grid_search

st.set_page_config(
    page_title="Retail Risk Monitor",
    page_icon="🛒",
    layout="wide",
)

CONFIG_PATH = str(PROJECT_ROOT / "configs" / "store_layout.yaml")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stHeader"]           { background-color: #0d1117; }
h1, h2, h3 { color: #e6edf3 !important; }

.metric-row {
    display: flex;
    gap: 10px;
    margin-bottom: 14px;
}
.metric-card {
    flex: 1;
    background: #161b22;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid #30363d;
    min-width: 0;
}
.metric-card.green  { border-left-color: #3fb950; }
.metric-card.orange { border-left-color: #d29922; }
.metric-card.red    { border-left-color: #f85149; }
.metric-card.blue   { border-left-color: #388bfd; }
.metric-label {
    font-size: 0.70rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 2px;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.1;
}

.alert-card {
    background: #161b22;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-left: 4px solid #30363d;
}
.alert-card.HIGH   { border-left-color: #f85149; background: #1e0d0d; }
.alert-card.MEDIUM { border-left-color: #d29922; background: #1c1800; }
.alert-card.LOW    { border-left-color: #388bfd; background: #0d1a2d; }
.alert-title  { font-weight: 700; color: #e6edf3; margin-bottom: 4px; font-size: 0.9rem; }
.alert-reason { color: #8b949e; font-size: 0.80rem; padding-left: 6px; line-height: 1.6; }

.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid #21262d;
}

.no-alert { color: #3fb950; font-size: 0.85rem; padding: 10px 0; }
</style>
""", unsafe_allow_html=True)


def _score_to_bgr(score: float):
    if score < 0.35:
        return (45, 200, 55)
    elif score < 0.65:
        return (0, 160, 255)
    else:
        return (35, 35, 230)


_ZONE_BGR = {
    "entrance":       (60,  200,  60),
    "walkway":        (200, 200,  60),
    "shelves_left":   (255, 130,  60),
    "shelves_center": (255, 100, 100),
    "shelves_right":  (220,  60,  60),
    "billing":        (200,  60, 200),
    "exit":           (60,  200, 200),
}

_PALETTE = [
    (100, 200, 255), (255, 200, 100), (180, 255, 100),
    (255, 100, 180), (100, 180, 255), (255, 180, 60),
    (60,  255, 180), (200, 100, 255), (100, 255, 200),
]


def _zone_colour(name: str, idx: int) -> tuple:
    for key, col in _ZONE_BGR.items():
        if key in name or name in key:
            return col
    return _PALETTE[idx % len(_PALETTE)]


_ALERT_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🔵", "NONE": "⚪"}


def _draw_zone_overlay(frame: np.ndarray, zone_mapper: ZoneMapper) -> None:
    overlay = frame.copy()
    for idx, (zone_name, polygon) in enumerate(zone_mapper._zones.items()):
        pts = polygon.astype(np.int32).reshape((-1, 1, 2))
        colour = _zone_colour(zone_name, idx)
        cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    for idx, (zone_name, polygon) in enumerate(zone_mapper._zones.items()):
        pts = polygon.astype(np.int32)
        colour = _zone_colour(zone_name, idx)
        cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, colour, 1, cv2.LINE_AA)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        short = zone_name.replace("shelves_", "shv_")
        cv2.putText(frame, short, (cx - 18, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 230), 1, cv2.LINE_AA)


def _annotate_frame(
    frame: np.ndarray,
    tracked_persons,
    scores: dict,
    zone_mapper: ZoneMapper | None,
    scale: int = 2,
) -> np.ndarray:
    out = frame.copy()

    if zone_mapper is not None:
        _draw_zone_overlay(out, zone_mapper)

    for person in tracked_persons:
        x, y, w, h = person.bbox
        score = scores.get(person.id, 0.0)
        colour = _score_to_bgr(score)

        fill_layer = out.copy()
        cv2.rectangle(fill_layer, (x, y), (x + w, y + h), colour, -1)
        cv2.addWeighted(fill_layer, 0.18, out, 0.82, 0, out)

        thickness = 3 if score >= 0.65 else 2
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"ID {person.id}  {score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, font, 0.45, 1)
        lx = x
        ly = max(y - lh - 8, 0)
        cv2.rectangle(out, (lx, ly), (lx + lw + 6, ly + lh + 7), colour, -1)
        cv2.putText(out, label, (lx + 3, ly + lh + 2),
                    font, 0.45, (12, 12, 12), 1, cv2.LINE_AA)

        zone_label = person.zone or ""
        if zone_label and zone_label != "unknown":
            (zw, zh), _ = cv2.getTextSize(zone_label, font, 0.32, 1)
            zx, zy = x, y + h + 3
            cv2.rectangle(out, (zx, zy), (zx + zw + 4, zy + zh + 5), colour, -1)
            cv2.putText(out, zone_label, (zx + 2, zy + zh + 2),
                        font, 0.32, (12, 12, 12), 1, cv2.LINE_AA)

        fx, fy = int(x + w / 2), int(y + h)
        cv2.circle(out, (fx, fy), 5, colour, -1)
        cv2.circle(out, (fx, fy), 5, (240, 240, 240), 1, cv2.LINE_AA)

    h_out, w_out = out.shape[:2]
    out = cv2.resize(out, (w_out * scale, h_out * scale), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _update_heatmap(heatmap: np.ndarray, tracked_persons) -> np.ndarray:
    for person in tracked_persons:
        x, y, w, h = person.bbox
        fx, fy = int(x + w / 2), int(y + h)
        if 0 <= fy < heatmap.shape[0] and 0 <= fx < heatmap.shape[1]:
            heatmap[fy, fx] += 1
    return heatmap


def _render_heatmap(heatmap: np.ndarray, scale: int = 2) -> np.ndarray:
    if heatmap.max() == 0:
        return np.zeros((heatmap.shape[0] * scale, heatmap.shape[1] * scale, 3), dtype=np.uint8)
    blur = cv2.GaussianBlur(heatmap, (21, 21), 0)
    norm = (blur / blur.max() * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    h, w = coloured.shape[:2]
    return cv2.resize(coloured, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)


def _metrics_html(frame_idx: int, n_persons: int, n_alerts: int, max_score: float) -> str:
    risk_class = "red" if max_score >= 0.65 else ("orange" if max_score >= 0.35 else "green")
    return f"""
<div class="metric-row">
  <div class="metric-card blue">
    <div class="metric-label">Frame</div>
    <div class="metric-value">{frame_idx}</div>
  </div>
  <div class="metric-card green">
    <div class="metric-label">Tracked</div>
    <div class="metric-value">{n_persons}</div>
  </div>
  <div class="metric-card orange">
    <div class="metric-label">Alerts</div>
    <div class="metric-value">{n_alerts}</div>
  </div>
  <div class="metric-card {risk_class}">
    <div class="metric-label">Peak Risk</div>
    <div class="metric-value">{max_score:.2f}</div>
  </div>
</div>
"""


def _alerts_html(alerts: list) -> str:
    if not alerts:
        return '<div class="no-alert">✅ No active alerts</div>'
    html = ""
    for pid, level, reasons in alerts:
        icon = _ALERT_ICON.get(level, "⚪")
        html += f'<div class="alert-card {level}">'
        html += f'<div class="alert-title">{icon} Person {pid} - {level}</div>'
        for r in reasons[:3]:
            html += f'<div class="alert-reason">• {r}</div>'
        html += "</div>"
    return html


def _zone_draw_ui(first_frame: np.ndarray) -> None:
    if not _HAS_CANVAS:
        st.error("streamlit-drawable-canvas not installed. Run: pip install streamlit-drawable-canvas")
        return

    import base64 as _b64mod, io as _iomod

    h, w = first_frame.shape[:2]

    canvas_w = min(700, w * 2) if w < 500 else min(700, w)
    canvas_scale = canvas_w / w
    canvas_h = int(h * canvas_scale)

    frame_rgb  = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    frame_disp = cv2.resize(frame_rgb, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    buf = _iomod.BytesIO()
    _PILImage.fromarray(frame_disp).convert("RGB").save(buf, format="JPEG", quality=75)
    data_url = "data:image/jpeg;base64," + _b64mod.b64encode(buf.getvalue()).decode()

    video_name = st.session_state.get("_video_name", "default")
    canvas_key = f"zone_canvas_{abs(hash(video_name)) % 10_000_000}"

    initial_drawing = {
        "version": "5.3.0",
        "objects": [],
        "backgroundImage": {
            "type":       "image",
            "version":    "5.3.0",
            "originX":    "left",
            "originY":    "top",
            "left":       0,
            "top":        0,
            "width":      canvas_w,
            "height":     canvas_h,
            "scaleX":     1.0,
            "scaleY":     1.0,
            "angle":      0,
            "opacity":    1,
            "visible":    True,
            "src":        data_url,
            "crossOrigin": None,
            "filters":    [],
        },
    }

    st.markdown(
        "**Drag rectangles on the frame to define each store zone, "
        "then name them below.**"
    )
    st.caption(
        "Use `billing` / `exit` exactly so bypass-detection works. "
        "Common names: `shelves`, `walkway`, `entrance`."
    )

    canvas_result = st_canvas(
        fill_color="rgba(255, 140, 0, 0.20)",
        stroke_width=2,
        stroke_color="#00ff88",
        background_color="",
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",
        initial_drawing=initial_drawing,
        key=canvas_key,
    )

    shapes = []
    if canvas_result.json_data is not None:
        shapes = [s for s in canvas_result.json_data.get("objects", [])
                  if s.get("type") == "rect"]

    if not shapes:
        st.info("Draw at least one rectangle above, then name it and click **Confirm zones**.")
        return

    _ZONE_OPTIONS = [
        "- select zone -",
        "billing",
        "exit",
        "shelves",
        "shelves_left",
        "shelves_center",
        "shelves_right",
        "walkway",
        "entrance",
        "other (type below)",
    ]

    st.markdown(f"**{len(shapes)} zone(s) drawn - name each one:**")
    n_cols = min(len(shapes), 4)
    cols = st.columns(n_cols)
    zone_names = []
    for i in range(len(shapes)):
        with cols[i % n_cols]:
            choice = st.selectbox(f"Zone {i + 1}", _ZONE_OPTIONS,
                                  key=f"zone_sel_{i}")
            if choice == "other (type below)":
                custom = st.text_input("Custom name", key=f"zone_custom_{i}",
                                       placeholder="e.g. storage")
                zone_names.append(custom.strip().lower())
            elif choice == "- select zone -":
                zone_names.append("")
            else:
                zone_names.append(choice)

    if st.button("✅ Confirm zones", type="primary"):
        missing = [i + 1 for i, n in enumerate(zone_names) if not n]
        if missing:
            st.error(f"Please name zone(s): {missing}")
            return

        custom_zones: dict[str, np.ndarray] = {}
        for shape, name in zip(shapes, zone_names):
            left = shape["left"]
            top  = shape["top"]
            sw   = shape["width"]  * shape.get("scaleX", 1.0)
            sh   = shape["height"] * shape.get("scaleY", 1.0)
            x1 = left / canvas_scale
            y1 = top  / canvas_scale
            x2 = (left + sw) / canvas_scale
            y2 = (top  + sh) / canvas_scale
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            custom_zones[name] = pts

        st.session_state["custom_zones"] = custom_zones
        video_name_save = st.session_state.get("_video_name", "")
        st.session_state["zones_video"]  = video_name_save
        st.session_state.pop("_is_redrawing", None)

        if video_name_save:
            import yaml as _yaml
            stem = Path(video_name_save).stem
            zone_cfg = PROJECT_ROOT / "configs" / f"zones_{stem}.yaml"
            with open(zone_cfg, "w") as _zf:
                _yaml.dump({n: v.tolist() for n, v in custom_zones.items()}, _zf)

        st.rerun()


def _extract_first_frame(uploaded) -> tuple[np.ndarray | None, str]:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name
    cap = cv2.VideoCapture(path)
    frame = None
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            frame = None
    cap.release()
    uploaded.seek(0)
    return frame, path


def _show_evaluation(all_final_features: dict, video_stem: str) -> None:
    import json as _json
    import matplotlib.pyplot as _plt

    st.divider()
    st.markdown('<div class="section-label">Model Training Results - Evaluation & Optimisation</div>', unsafe_allow_html=True)
    st.caption(
        "Upload a ground truth JSON file (see `data/ground_truth_sample.json`) "
        "to compute accuracy, precision, recall, F1 and find optimal scoring weights. "
        "Person IDs match those in the Final Summary table above."
    )
    gt_file = st.file_uploader("Ground truth JSON", type=["json"], key="gt_upload_phase2")
    if not gt_file:
        return

    gt_data = _json.load(gt_file)
    gt_for_video = gt_data.get("videos", {}).get(video_stem, {})

    if not gt_for_video:
        st.warning(
            f"No annotations for **'{video_stem}'** in this file. "
            f"Available: `{list(gt_data.get('videos', {}).keys())}`"
        )
        return

    ground_truth = {int(k): v for k, v in gt_for_video.items()}
    common_pids  = [pid for pid in all_final_features if pid in ground_truth]

    if not common_pids:
        st.warning(
            "No overlap between tracked IDs "
            f"{sorted(all_final_features.keys())} and GT IDs "
            f"{sorted(ground_truth.keys())}."
        )
        return

    y_true = [ground_truth[pid] for pid in common_pids]
    y_pred = [get_alert_level(all_final_features[pid][0]) for pid in common_pids]
    report = classification_report(y_true, y_pred)

    accuracy   = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1   = report.get("macro avg", {}).get("f1",        0.0)
    macro_prec = report.get("macro avg", {}).get("precision", 0.0)
    macro_rec  = report.get("macro avg", {}).get("recall",    0.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",         f"{accuracy:.1%}")
    m2.metric("Macro F1",         f"{macro_f1:.3f}")
    m3.metric("Macro Precision",  f"{macro_prec:.3f}")
    m4.metric("Macro Recall",     f"{macro_rec:.3f}")

    st.divider()

    col_rep, col_cm = st.columns(2)

    with col_rep:
        st.markdown("**Classification Report**")
        st.dataframe(
            pd.DataFrame([
                {"Class": c,
                 "Precision": f"{v['precision']:.3f}",
                 "Recall":    f"{v['recall']:.3f}",
                 "F1":        f"{v['f1']:.3f}",
                 "Support":   v["support"]}
                for c, v in report.items()
            ]),
            use_container_width=True, hide_index=True,
        )

    with col_cm:
        st.markdown("**Confusion Matrix**")
        LEVELS  = ["NONE", "LOW", "MEDIUM", "HIGH"]
        classes = [l for l in LEVELS if l in set(y_true) | set(y_pred)]
        n       = len(classes)
        idx     = {c: i for i, c in enumerate(classes)}
        cm      = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[idx[t]][idx[p]] += 1

        fig_cm, ax = _plt.subplots(figsize=(4, 3))
        fig_cm.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
        ax.set_xticks(range(n));  ax.set_xticklabels(classes, color="white", fontsize=9)
        ax.set_yticks(range(n));  ax.set_yticklabels(classes, color="white", fontsize=9)
        ax.set_xlabel("Predicted", color="#8b949e", fontsize=9)
        ax.set_ylabel("True",      color="#8b949e", fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        threshold = cm.max() * 0.55
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="#0d1117" if cm[i, j] > threshold else "white")
        _plt.tight_layout(pad=0.5)
        st.pyplot(fig_cm)
        _plt.close(fig_cm)

    st.divider()

    st.markdown("**Weight Grid Search** - optimises (α, β, γ, δ) to maximise macro-F1")
    st.caption("α = dwell anomaly  ·  β = zone revisits  ·  γ = path irregularity  ·  δ = billing bypass  ·  286 combinations, step = 0.1, α+β+γ+δ = 1")

    if st.button("▶  Run grid search", key="run_grid_search"):
        with st.spinner("Searching 286 weight combinations…"):
            best, all_results = run_grid_search(all_final_features, ground_truth, step=0.1)

        if not best:
            st.warning("No results - check ID overlap.")
            return

        st.success(
            f"Best weights found:  α = {best['alpha']}  β = {best['beta']} "
            f" γ = {best['gamma']}  δ = {best['delta']}  →  macro-F1 = **{best['macro_f1']:.3f}**"
        )

        col_curve, col_top = st.columns(2)

        with col_curve:
            st.markdown("**Optimisation Curve**")
            st.caption("Each point = one weight combination. Running-max line shows the best F1 found so far as the search progresses.")

            f1_vals = [r["macro_f1"] for r in reversed(all_results)]
            running_max, cur = [], 0.0
            for v in f1_vals:
                cur = max(cur, v)
                running_max.append(cur)

            curve_df = pd.DataFrame({
                "F1":              f1_vals,
                "Best F1 so far":  running_max,
            }, index=range(1, len(f1_vals) + 1))
            curve_df.index.name = "Combination #"
            st.line_chart(curve_df, height=240)

        with col_top:
            st.markdown("**Top 10 Weight Combinations**")
            top10 = pd.DataFrame(all_results[:10]).rename(columns={
                "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "macro_f1": "F1"
            })
            st.dataframe(top10, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("🛒 Retail Risk Monitor")

    with st.sidebar:
        st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a video file", type=["mpg", "mpeg", "mp4", "avi"]
        )
        process_all = st.checkbox("Process full video", value=True)
        if not process_all:
            max_frames = st.slider("Max frames (for quick testing)", 50, 2000, 200, step=50)
        else:
            max_frames = 999_999
        conf_threshold = st.slider("Detection confidence", 0.1, 0.9, 0.4, step=0.05)
        max_age = st.slider(
            "Track memory (frames)",
            min_value=5, max_value=150, value=30, step=5,
            help="Frames a lost track is kept alive before being deleted. 30 @ 25 fps ≈ 1.2 s.",
        )
        iou_threshold = st.slider(
            "Match score threshold",
            min_value=0.05, max_value=0.6, value=0.25, step=0.05,
            help="Minimum combined IoU+appearance score to match a detection to an existing track. Raise if IDs still swap; lower if tracks are lost.",
        )
        st.markdown('<div class="section-label">Zone Layout</div>', unsafe_allow_html=True)
        zone_mode = st.radio(
            "Zone mode",
            [
                "Draw custom zones",
                "Auto (adapt to video)",
                "CAVIAR config (384×288)",
                "Disabled",
            ],
            help=(
                "Draw custom zones: interactively place zone boxes on the first frame.\n"
                "Auto: proportional grid, works for any video.\n"
                "CAVIAR config: fixed polygons for 384×288 CAVIAR footage only.\n"
                "Disabled: turns off all zone features."
            ),
        )
        upscale = st.toggle("2× upscale display", value=True)
        save_video = st.toggle("Save annotated video", value=False,
                               help="Write the annotated output to data/output/ as an MP4 file.")
        detect_every = st.slider(
            "Process every N frames",
            min_value=1, max_value=6, value=2, step=1,
            help="1 = every frame (slowest). 2 = 2× faster. 3 = 3× faster. YOLO is skipped on intermediate frames; Kalman filter predicts positions instead.",
        )
        st.divider()
        st.markdown('<div class="section-label">Calibration Status</div>', unsafe_allow_html=True)
        calib_ph = st.empty()
        st.divider()
        run_btn = st.button("▶  Run pipeline", type="primary", use_container_width=True)

    if not uploaded:
        st.info("Upload a video file in the sidebar and press **Run pipeline** to start.")
        return

    st.session_state["_video_name"] = uploaded.name
    saved_zones_video = st.session_state.get("zones_video", "")
    if saved_zones_video and saved_zones_video != uploaded.name:
        st.session_state.pop("custom_zones", None)
        st.session_state.pop("zones_video", None)

    if zone_mode == "Draw custom zones" and not st.session_state.get("custom_zones") \
            and not st.session_state.get("_is_redrawing"):
        import yaml as _yaml
        _stem = Path(uploaded.name).stem
        _zone_cfg = PROJECT_ROOT / "configs" / f"zones_{_stem}.yaml"
        if _zone_cfg.exists():
            with open(_zone_cfg) as _zf:
                _raw = _yaml.safe_load(_zf) or {}
            if _raw:
                st.session_state["custom_zones"] = {n: np.array(v, dtype=np.float32) for n, v in _raw.items()}
                st.session_state["zones_video"] = uploaded.name

    if not run_btn:
        first_frame, _tmp_path = _extract_first_frame(uploaded)

        if first_frame is None:
            st.error("Could not read the first frame of the video.")
            return

        ph, pw = first_frame.shape[:2]
        scale_prev = 2 if upscale else 1

        if zone_mode == "Draw custom zones":
            confirmed = st.session_state.get("custom_zones")
            confirmed_video = st.session_state.get("zones_video", "")

            if confirmed and confirmed_video == uploaded.name:
                zm_prev = ZoneMapper.from_dict(confirmed)
                preview = first_frame.copy()
                _draw_zone_overlay(preview, zm_prev)
                preview = cv2.resize(
                    preview, (pw * scale_prev, ph * scale_prev), interpolation=cv2.INTER_LINEAR
                )
                st.markdown('<div class="section-label">Custom Zones - confirmed</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                zone_list = ", ".join(f"`{z}`" for z in confirmed.keys())
                st.success(f"Zones confirmed: {zone_list}")
                col_ok, col_reset = st.columns([3, 1])
                with col_reset:
                    if st.button("Redraw zones"):
                        st.session_state.pop("custom_zones", None)
                        st.session_state.pop("zones_video", None)
                        st.session_state["_is_redrawing"] = True
                        st.rerun()
                with col_ok:
                    st.info("Press **▶ Run pipeline** in the sidebar to start.")
            else:
                st.markdown('<div class="section-label">Draw Zones - first frame</div>', unsafe_allow_html=True)
                _zone_draw_ui(first_frame)

            _ev_features = st.session_state.get("_pipeline_features", {})
            if _ev_features and st.session_state.get("_pipeline_video") == uploaded.name:
                _show_evaluation(_ev_features, Path(uploaded.name).stem)
            return

        if zone_mode == "Auto (adapt to video)":
            zm_prev = ZoneMapper.from_frame(ph, pw)
        elif zone_mode == "CAVIAR config (384×288)":
            zm_prev = ZoneMapper(CONFIG_PATH)
        else:
            zm_prev = None

        preview = first_frame.copy()
        if zm_prev is not None:
            _draw_zone_overlay(preview, zm_prev)
        preview = cv2.resize(
            preview, (pw * scale_prev, ph * scale_prev), interpolation=cv2.INTER_LINEAR
        )
        st.markdown('<div class="section-label">Zone Preview - first frame</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if zm_prev is None:
            st.info("Zones disabled - zone-based features will be skipped.")
        elif zone_mode == "CAVIAR config (384×288)" and (ph != 288 or pw != 384):
            st.warning(f"Video is {pw}×{ph} but CAVIAR config expects 384×288 - zones will be misaligned. Switch to **Auto** or **Draw custom zones**.")
        st.info("Zones look correct? Press **▶ Run pipeline** in the sidebar to start.")

        _ev_features = st.session_state.get("_pipeline_features", {})
        if _ev_features and st.session_state.get("_pipeline_video") == uploaded.name:
            _show_evaluation(_ev_features, Path(uploaded.name).stem)
        return

    if zone_mode == "Draw custom zones":
        confirmed = st.session_state.get("custom_zones")
        confirmed_video = st.session_state.get("zones_video", "")
        if not confirmed or confirmed_video != uploaded.name:
            st.error("No zones confirmed yet. Please draw zones on the first frame first (Run button will activate after confirming).")
            return

    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open the video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 288
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 384
    scale = 2 if upscale else 1

    if zone_mode == "Draw custom zones":
        zone_mapper = ZoneMapper.from_dict(st.session_state["custom_zones"])
    elif zone_mode == "Auto (adapt to video)":
        zone_mapper = ZoneMapper.from_frame(frame_h, frame_w)
    elif zone_mode == "CAVIAR config (384×288)":
        zone_mapper = ZoneMapper(CONFIG_PATH)
    else:
        zone_mapper = None

    show_zones = zone_mapper is not None

    detector = PersonDetector(model_name="yolov8n.pt")
    detector._model.overrides["conf"] = conf_threshold
    tracker = PersonTracker(max_age=max_age, iou_threshold=iou_threshold)
    zone_graph = ZoneTransitionGraph()
    behavior_tracker = BehaviorTracker(fps=fps)
    adaptive_scorer = AdaptiveScorer()

    shap_explainer = SHAPExplainer()
    all_final_features: dict[int, tuple] = {}

    # Set up a video writer if the user wants to save annotated output.
    writer = None
    output_path = None
    if save_video:
        out_dir = PROJECT_ROOT / "data" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(uploaded.name).stem
        output_path = out_dir / f"{stem}_annotated.mp4"
        out_w = frame_w * scale
        out_h = frame_h * scale
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps / detect_every, (out_w, out_h))

    metrics_ph = st.empty()

    col_video, col_alerts = st.columns([3, 2])
    with col_video:
        st.markdown('<div class="section-label">Live Feed</div>', unsafe_allow_html=True)
        video_ph = st.empty()
    with col_alerts:
        st.markdown('<div class="section-label">Active Alerts</div>', unsafe_allow_html=True)
        alerts_ph = st.empty()

    st.divider()
    col_scores, col_heatmap = st.columns(2)
    with col_scores:
        st.markdown('<div class="section-label">Suspicion Scores</div>', unsafe_allow_html=True)
        scores_ph = st.empty()
    with col_heatmap:
        st.markdown('<div class="section-label">Foot-Position Heatmap</div>', unsafe_allow_html=True)
        heatmap_ph = st.empty()

    st.divider()
    st.markdown('<div class="section-label">Risk Score Timeline</div>', unsafe_allow_html=True)
    timeline_ph = st.empty()

    st.divider()
    st.markdown('<div class="section-label">XAI - Live Risk Drivers (SHAP)</div>', unsafe_allow_html=True)
    st.caption("Which behavioural features are pushing each person's risk above or below the crowd average.")
    shap_live_ph = st.empty()

    heatmap_acc = np.zeros((frame_h, frame_w), dtype=np.float32)
    score_history: dict = {}
    frame_history: dict = {}
    score_breakdowns: dict = {}
    progress = st.progress(0, text="Processing…")

    _PANEL_EVERY = detect_every * 4

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        run_detect = (frame_idx % detect_every == 0)
        detections = detector.detect(frame) if run_detect else []
        tracked_persons = tracker.update(detections, frame, frame_idx, fps)

        if show_zones:
            for person in tracked_persons:
                person.zone = zone_mapper.get_zone(person.bbox)

        heatmap_acc = _update_heatmap(heatmap_acc, tracked_persons)

        current_scores: dict = {}
        current_alert_levels: dict = {}
        current_alerts: list = []

        for person in tracked_persons:
            features = behavior_tracker.update(person, zone_graph)
            features, breakdown = adaptive_scorer.compute(features)
            features = generate_alert(features)
            pid = person.id
            level = get_alert_level(features)
            current_scores[pid] = features.suspicion_score
            current_alert_levels[pid] = level
            score_breakdowns[pid] = breakdown
            raw = breakdown.get("raw", [0.0, 0.0, 0.0, 0.0])
            shap_explainer.update(pid, raw)
            all_final_features[pid] = (features, raw)
            if pid not in score_history:
                score_history[pid] = []
                frame_history[pid] = []
            score_history[pid].append(features.suspicion_score)
            frame_history[pid].append(frame_idx)
            if level in ("MEDIUM", "HIGH"):
                current_alerts.append((pid, level, features.alert_reasons))

        if run_detect:
            annotated_rgb = _annotate_frame(
                frame, tracked_persons, current_scores,
                zone_mapper=zone_mapper if show_zones else None,
                scale=scale,
            )
            video_ph.image(annotated_rgb, channels="RGB", use_container_width=True)

            # Write frame to file if video saving is on.
            if writer is not None:
                writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

            max_score = max(current_scores.values(), default=0.0)
            metrics_ph.markdown(
                _metrics_html(frame_idx, len(tracked_persons), len(current_alerts), max_score),
                unsafe_allow_html=True,
            )

        if frame_idx % _PANEL_EVERY == 0:
            alerts_ph.markdown(_alerts_html(current_alerts), unsafe_allow_html=True)

            if current_scores:
                _LEVEL_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "NONE": "🟢"}
                rows = []
                for pid, s in sorted(current_scores.items()):
                    bd  = score_breakdowns.get(pid, {})
                    lvl = current_alert_levels.get(pid, "NONE")
                    rows.append({
                        "Person":    f"ID {pid}",
                        "Score":     f"{s:.3f}",
                        "Alert":     f"{_LEVEL_ICON.get(lvl, '')} {lvl}",
                        "Dwell":     bd.get("Dwell anomaly", 0),
                        "Revisits":  bd.get("Zone revisits", 0),
                        "Irregular": bd.get("Path irregularity", 0),
                        "Bypass":    bd.get("Billing bypass", 0),
                    })
                scores_ph.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            th = adaptive_scorer.current_thresholds()
            mode_str = "Adaptive ✅" if th["adaptive"] else f"Fixed (need {AdaptiveScorer._MIN_SAMPLES}+ persons)"
            calib_ph.markdown(
                f"**Mode:** {mode_str}<br>"
                f"**Persons seen:** {th['n']}<br>"
                f"**Dwell threshold:** {th['dwell_s']}s<br>"
                f"**Revisit threshold:** {th['revisits']}",
                unsafe_allow_html=True,
            )

            heatmap_ph.image(
                _render_heatmap(heatmap_acc, scale=scale),
                channels="RGB", use_container_width=True,
            )

            if score_history:
                timeline_df = pd.DataFrame({
                    f"P{pid}": pd.Series(score_history[pid], index=frame_history[pid])
                    for pid in score_history
                })
                timeline_ph.line_chart(timeline_df, height=180)

            import matplotlib.pyplot as _plt
            live_exp = shap_explainer.explain_all()
            if live_exp:
                _fig_live = plot_summary_bar(live_exp)
                shap_live_ph.pyplot(_fig_live)
                _plt.close(_fig_live)

        pct = int((frame_idx + 1) / max_frames * 100)
        progress.progress(
            min(pct, 100),
            text=f"Frame {frame_idx + 1} / {min(max_frames, total_video_frames)}",
        )
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        st.success(f"Annotated video saved to `{output_path}`")
    progress.progress(100, text="Done.")

    st.divider()
    st.markdown('<div class="section-label">Final Summary</div>', unsafe_allow_html=True)
    all_features = behavior_tracker.all_features()
    if all_features:
        rows = []
        for pid, feat in sorted(all_features.items()):
            feat, bd = adaptive_scorer.compute_final(feat)
            feat = generate_alert(feat)
            raw_final = bd.get("raw", all_final_features.get(pid, (None, [0.0, 0.0, 0.0, 0.0]))[1])
            all_final_features[pid] = (feat, raw_final)
            shap_explainer.update(pid, raw_final)
            level = get_alert_level(feat)
            rows.append({
                "Person":           f"ID {pid}",
                "Zones Visited":    ", ".join(dict.fromkeys(feat.zone_sequence)) or "-",
                "Revisits":         sum(feat.zone_revisits.values()),
                "Billing Bypassed": "✅ YES" if feat.billing_bypassed else "no",
                "Score":            f"{feat.suspicion_score:.3f}",
                "Dwell (+pts)":     bd.get("Dwell anomaly", 0),
                "Revisit (+pts)":   bd.get("Zone revisits", 0),
                "Irregular (+pts)": bd.get("Path irregularity", 0),
                "Billing (+pts)":   bd.get("Billing bypass", 0),
                "Alert":            f"{_ALERT_ICON.get(level, '')} {level}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No persons were tracked in the processed frames.")

    st.session_state["_pipeline_features"] = all_final_features
    st.session_state["_pipeline_video"]    = uploaded.name

    st.divider()
    col_hm, col_shap_sum = st.columns(2)
    with col_hm:
        st.markdown('<div class="section-label">Final Heatmap</div>', unsafe_allow_html=True)
        st.image(_render_heatmap(heatmap_acc, scale=scale), channels="RGB", use_container_width=True)
    with col_shap_sum:
        st.markdown('<div class="section-label">XAI - Risk Drivers (SHAP)</div>', unsafe_allow_html=True)
        import matplotlib.pyplot as _plt
        explanations = shap_explainer.explain_all()
        if explanations:
            _fig_sum = plot_summary_bar(explanations)
            st.pyplot(_fig_sum)
            _plt.close(_fig_sum)
            st.caption("Mean |SHAP value| per feature across all tracked persons. Shows which behavioural signal contributes most to risk in this scene.")
        else:
            st.info("Need 2+ tracked persons to compute SHAP baseline.")

    if explanations:
        st.divider()
        st.markdown('<div class="section-label">XAI - Per-Person Attribution (SHAP Waterfall)</div>', unsafe_allow_html=True)
        st.caption("Red = feature pushed this person's risk ABOVE the crowd average. Green = below average. Base value = population mean risk.")
        n_cols = min(len(explanations), 3)
        wf_cols = st.columns(n_cols)
        for i, (pid, exp) in enumerate(sorted(explanations.items())):
            with wf_cols[i % n_cols]:
                _fig_wf = plot_waterfall(exp, pid)
                st.pyplot(_fig_wf)
                _plt.close(_fig_wf)

    _show_evaluation(all_final_features, Path(uploaded.name).stem)


if __name__ == "__main__":
    main()
