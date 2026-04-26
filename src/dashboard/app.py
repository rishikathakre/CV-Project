"""
Streamlit real-time retail risk dashboard.

Upload a .mpg (or any OpenCV-readable) video via the sidebar.
The full pipeline runs frame-by-frame and displays:
  - Smooth annotated live feed (every frame, 2× upscaled)
  - Score-based bounding box colors + semi-transparent fill
  - Zone overlay with colored region polygons
  - Per-person suspicion score table with risk indicators
  - Active alerts panel with severity cards
  - Gaussian-blurred foot-position heatmap
  - Risk score timeline chart per person
  - Metric cards: frame / tracked / alerts / peak risk
"""

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
    # streamlit-drawable-canvas 0.9.3 calls st_image.image_to_url which was
    # removed in Streamlit 1.25+.  Inject a shim before importing the component.
    import base64 as _b64, io as _io, streamlit.elements.image as _st_img
    if not hasattr(_st_img, "image_to_url"):
        def _image_to_url(image, width, clamp, channels, output_format, image_id,
                          allow_emoji=False):
            buf = _io.BytesIO()
            img = image if hasattr(image, "save") else _PILImage.fromarray(image)
            # Use JPEG at quality 80 — much smaller payload than PNG, passes
            # through Streamlit's component protocol without truncation.
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
# Custom CSS — dark professional theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stHeader"]           { background-color: #0d1117; }
h1, h2, h3 { color: #e6edf3 !important; }

/* ---- Metric cards ---- */
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

/* ---- Alert cards ---- */
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

/* ---- Section label ---- */
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

/* ---- No-alert placeholder ---- */
.no-alert { color: #3fb950; font-size: 0.85rem; padding: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _score_to_bgr(score: float):
    """Green → amber → red based on suspicion score."""
    if score < 0.35:
        return (45, 200, 55)     # green  (BGR)
    elif score < 0.65:
        return (0, 160, 255)     # orange (BGR)
    else:
        return (35, 35, 230)     # red    (BGR)


# Named-zone colours; any unknown zone name gets a colour from the palette.
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
    # Try substrings for common names
    for key, col in _ZONE_BGR.items():
        if key in name or name in key:
            return col
    return _PALETTE[idx % len(_PALETTE)]


_ALERT_ICON = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🔵", "NONE": "⚪"}

# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def _draw_zone_overlay(frame: np.ndarray, zone_mapper: ZoneMapper) -> None:
    """In-place: semi-transparent zone fills, outlines, and short labels."""
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
    """
    Draw zone overlay (optional), score-coloured boxes with semi-transparent
    fill, solid-background labels, zone tags, and foot-position dots.
    Returns an upscaled RGB image ready for st.image().
    """
    out = frame.copy()

    if zone_mapper is not None:
        _draw_zone_overlay(out, zone_mapper)

    for person in tracked_persons:
        x, y, w, h = person.bbox
        score = scores.get(person.id, 0.0)
        colour = _score_to_bgr(score)

        # Semi-transparent fill
        fill_layer = out.copy()
        cv2.rectangle(fill_layer, (x, y), (x + w, y + h), colour, -1)
        cv2.addWeighted(fill_layer, 0.18, out, 0.82, 0, out)

        # Box outline — thicker at high risk
        thickness = 3 if score >= 0.65 else 2
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness, cv2.LINE_AA)

        # --- Top label: "ID 2  0.71" ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"ID {person.id}  {score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, font, 0.45, 1)
        lx = x
        ly = max(y - lh - 8, 0)
        cv2.rectangle(out, (lx, ly), (lx + lw + 6, ly + lh + 7), colour, -1)
        cv2.putText(out, label, (lx + 3, ly + lh + 2),
                    font, 0.45, (12, 12, 12), 1, cv2.LINE_AA)

        # --- Bottom zone tag ---
        zone_label = person.zone or ""
        if zone_label and zone_label != "unknown":
            (zw, zh), _ = cv2.getTextSize(zone_label, font, 0.32, 1)
            zx, zy = x, y + h + 3
            cv2.rectangle(out, (zx, zy), (zx + zw + 4, zy + zh + 5), colour, -1)
            cv2.putText(out, zone_label, (zx + 2, zy + zh + 2),
                        font, 0.32, (12, 12, 12), 1, cv2.LINE_AA)

        # Foot-position dot (white ring + colour fill)
        fx, fy = int(x + w / 2), int(y + h)
        cv2.circle(out, (fx, fy), 5, colour, -1)
        cv2.circle(out, (fx, fy), 5, (240, 240, 240), 1, cv2.LINE_AA)

    # Upscale for display
    h_out, w_out = out.shape[:2]
    out = cv2.resize(out, (w_out * scale, h_out * scale), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

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
        html += f'<div class="alert-title">{icon} Person {pid} — {level}</div>'
        for r in reasons[:3]:
            html += f'<div class="alert-reason">• {r}</div>'
        html += "</div>"
    return html


# ---------------------------------------------------------------------------
# Interactive zone drawing UI  (canvas rectangle drawing)
# ---------------------------------------------------------------------------

def _zone_draw_ui(first_frame: np.ndarray) -> None:
    """
    Drawable canvas on the first frame.  User drags rectangles over store zones,
    names each zone in the text boxes below, then clicks Confirm.
    Confirmed zones → st.session_state["custom_zones"].

    Background image is injected via initial_drawing["backgroundImage"] (fabric.js
    JSON) instead of the background_image parameter, which calls the removed
    Streamlit image_to_url API and causes a blank canvas.
    """
    if not _HAS_CANVAS:
        st.error("streamlit-drawable-canvas not installed. Run: pip install streamlit-drawable-canvas")
        return

    import base64 as _b64mod, io as _iomod

    h, w = first_frame.shape[:2]

    # Double small videos (CAVIAR 384 px) so they're usable; cap at 700 px
    canvas_w = min(700, w * 2) if w < 500 else min(700, w)
    canvas_scale = canvas_w / w
    canvas_h = int(h * canvas_scale)

    # Encode resized frame as a compact JPEG data URL
    frame_rgb  = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    frame_disp = cv2.resize(frame_rgb, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    buf = _iomod.BytesIO()
    _PILImage.fromarray(frame_disp).convert("RGB").save(buf, format="JPEG", quality=75)
    data_url = "data:image/jpeg;base64," + _b64mod.b64encode(buf.getvalue()).decode()

    # Embed the frame as a non-interactive fabric.js backgroundImage.
    # Using initial_drawing bypasses the broken image_to_url path in the
    # background_image parameter.  The useEffect that loads initial_drawing
    # only fires when the canvas element mounts (key change), so user-drawn
    # rectangles survive Streamlit reruns.
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
        background_color="",       # backgroundImage handles it
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
        "— select zone —",
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

    st.markdown(f"**{len(shapes)} zone(s) drawn — name each one:**")
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
            elif choice == "— select zone —":
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
        st.session_state["zones_video"]  = st.session_state.get("_video_name", "")
        st.rerun()


# ---------------------------------------------------------------------------
# Helpers for first-frame extraction
# ---------------------------------------------------------------------------

def _extract_first_frame(uploaded) -> tuple[np.ndarray | None, str]:
    """Write uploaded file to a temp path, read one frame, return (frame, path)."""
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
    # Rewind the uploader so it can be read again later
    uploaded.seek(0)
    return frame, path


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🛒 Retail Risk Monitor")

    # ---- Sidebar ----
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
            min_value=10, max_value=300, value=150, step=10,
            help="Frames a lost track is kept alive. 90 @ 25 fps ≈ 3.6 s.",
        )
        iou_threshold = st.slider(
            "IoU match threshold",
            min_value=0.01, max_value=0.5, value=0.10, step=0.01,
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
        st.divider()
        st.markdown('<div class="section-label">Calibration Status</div>', unsafe_allow_html=True)
        calib_ph = st.empty()
        st.divider()
        run_btn = st.button("▶  Run pipeline", type="primary", use_container_width=True)

    # ------------------------------------------------------------------ #
    # Nothing uploaded
    # ------------------------------------------------------------------ #
    if not uploaded:
        st.info("Upload a video file in the sidebar and press **Run pipeline** to start.")
        return

    # Track which video is loaded so we can invalidate saved zones if it changes
    st.session_state["_video_name"] = uploaded.name
    saved_zones_video = st.session_state.get("zones_video", "")
    if saved_zones_video and saved_zones_video != uploaded.name:
        # Different video uploaded — clear old custom zones
        st.session_state.pop("custom_zones", None)
        st.session_state.pop("zones_video", None)

    # ------------------------------------------------------------------ #
    # PRE-RUN PHASE  (Run button not pressed yet)
    # ------------------------------------------------------------------ #
    if not run_btn:
        first_frame, _tmp_path = _extract_first_frame(uploaded)

        if first_frame is None:
            st.error("Could not read the first frame of the video.")
            return

        ph, pw = first_frame.shape[:2]
        scale_prev = 2 if upscale else 1

        # ---- "Draw custom zones" mode ----
        if zone_mode == "Draw custom zones":
            confirmed = st.session_state.get("custom_zones")
            confirmed_video = st.session_state.get("zones_video", "")

            if confirmed and confirmed_video == uploaded.name:
                # Already confirmed — show a preview with the custom zones drawn
                zm_prev = ZoneMapper.from_dict(confirmed)
                preview = first_frame.copy()
                _draw_zone_overlay(preview, zm_prev)
                preview = cv2.resize(
                    preview, (pw * scale_prev, ph * scale_prev), interpolation=cv2.INTER_LINEAR
                )
                st.markdown('<div class="section-label">Custom Zones — confirmed</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                zone_list = ", ".join(f"`{z}`" for z in confirmed.keys())
                st.success(f"Zones confirmed: {zone_list}")
                col_ok, col_reset = st.columns([3, 1])
                with col_reset:
                    if st.button("Redraw zones"):
                        st.session_state.pop("custom_zones", None)
                        st.session_state.pop("zones_video", None)
                        st.rerun()
                with col_ok:
                    st.info("Press **▶ Run pipeline** in the sidebar to start.")
            else:
                # Drawing phase
                st.markdown('<div class="section-label">Draw Zones — first frame</div>', unsafe_allow_html=True)
                _zone_draw_ui(first_frame)

            return

        # ---- Other zone modes: static preview ----
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
        st.markdown('<div class="section-label">Zone Preview — first frame</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if zm_prev is None:
            st.info("Zones disabled — zone-based features will be skipped.")
        elif zone_mode == "CAVIAR config (384×288)" and (ph != 288 or pw != 384):
            st.warning(f"Video is {pw}×{ph} but CAVIAR config expects 384×288 — zones will be misaligned. Switch to **Auto** or **Draw custom zones**.")
        st.info("Zones look correct? Press **▶ Run pipeline** in the sidebar to start.")
        return

    # ------------------------------------------------------------------ #
    # PIPELINE PHASE  (Run button pressed)
    # ------------------------------------------------------------------ #

    # For "Draw custom zones": require zones to be confirmed first
    if zone_mode == "Draw custom zones":
        confirmed = st.session_state.get("custom_zones")
        confirmed_video = st.session_state.get("zones_video", "")
        if not confirmed or confirmed_video != uploaded.name:
            st.error("No zones confirmed yet. Please draw zones on the first frame first (Run button will activate after confirming).")
            return

    # ---- Save upload to temp file ----
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

    # ---- Zone mapper ----
    if zone_mode == "Draw custom zones":
        zone_mapper = ZoneMapper.from_dict(st.session_state["custom_zones"])
    elif zone_mode == "Auto (adapt to video)":
        zone_mapper = ZoneMapper.from_frame(frame_h, frame_w)
    elif zone_mode == "CAVIAR config (384×288)":
        zone_mapper = ZoneMapper(CONFIG_PATH)
    else:
        zone_mapper = None

    show_zones = zone_mapper is not None

    # ---- Pipeline components ----
    detector = PersonDetector(model_name="yolov8n.pt")
    detector._model.overrides["conf"] = conf_threshold
    tracker = PersonTracker(max_age=max_age, iou_threshold=iou_threshold)
    zone_graph = ZoneTransitionGraph()
    behavior_tracker = BehaviorTracker(fps=fps)
    adaptive_scorer = AdaptiveScorer()

    # ---- Layout ----
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

    # ---- State ----
    heatmap_acc = np.zeros((frame_h, frame_w), dtype=np.float32)
    score_history: dict = {}
    frame_history: dict = {}
    score_breakdowns: dict = {}
    progress = st.progress(0, text="Processing…")

    _PANEL_EVERY = 5

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked_persons = tracker.update(detections, frame, frame_idx, fps)

        if show_zones:
            for person in tracked_persons:
                person.zone = zone_mapper.get_zone(person.bbox)

        heatmap_acc = _update_heatmap(heatmap_acc, tracked_persons)

        current_scores: dict = {}
        current_alerts: list = []

        for person in tracked_persons:
            features = behavior_tracker.update(person, zone_graph)
            features, breakdown = adaptive_scorer.compute(features)
            features = generate_alert(features)
            pid = person.id
            current_scores[pid] = features.suspicion_score
            score_breakdowns[pid] = breakdown
            if pid not in score_history:
                score_history[pid] = []
                frame_history[pid] = []
            score_history[pid].append(features.suspicion_score)
            frame_history[pid].append(frame_idx)
            if get_alert_level(features) in ("MEDIUM", "HIGH"):
                current_alerts.append((pid, get_alert_level(features), features.alert_reasons))

        # ---- Video: update every frame ----
        annotated_rgb = _annotate_frame(
            frame, tracked_persons, current_scores,
            zone_mapper=zone_mapper if show_zones else None,
            scale=scale,
        )
        video_ph.image(annotated_rgb, channels="RGB", use_container_width=True)

        # ---- Metric cards: every frame ----
        max_score = max(current_scores.values(), default=0.0)
        metrics_ph.markdown(
            _metrics_html(frame_idx, len(tracked_persons), len(current_alerts), max_score),
            unsafe_allow_html=True,
        )

        # ---- Slower panels ----
        if frame_idx % _PANEL_EVERY == 0:
            alerts_ph.markdown(_alerts_html(current_alerts), unsafe_allow_html=True)

            if current_scores:
                def _risk_label(s):
                    if s >= 0.65:
                        return "🔴 HIGH"
                    if s >= 0.35:
                        return "🟠 MED"
                    return "🟢 LOW"

                rows = []
                for pid, s in sorted(current_scores.items()):
                    bd = score_breakdowns.get(pid, {})
                    rows.append({
                        "Person":        f"ID {pid}",
                        "Score":         f"{s:.3f}",
                        "Risk":          _risk_label(s),
                        "Dwell":         bd.get("Dwell anomaly", 0),
                        "Revisits":      bd.get("Zone revisits", 0),
                        "Irregular":     bd.get("Path irregularity", 0),
                        "BillingBypass": bd.get("Billing bypass", 0),
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

        pct = int((frame_idx + 1) / max_frames * 100)
        progress.progress(
            min(pct, 100),
            text=f"Frame {frame_idx + 1} / {min(max_frames, total_video_frames)}",
        )
        frame_idx += 1

    cap.release()
    progress.progress(100, text="Done.")

    # ---- Final summary ----
    st.divider()
    st.markdown('<div class="section-label">Final Summary</div>', unsafe_allow_html=True)
    all_features = behavior_tracker.all_features()
    if all_features:
        rows = []
        for pid, feat in sorted(all_features.items()):
            feat, bd = adaptive_scorer.compute_final(feat)
            feat = generate_alert(feat)
            level = get_alert_level(feat)
            rows.append({
                "Person":           f"ID {pid}",
                "Zones Visited":    ", ".join(dict.fromkeys(feat.zone_sequence)) or "—",
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

    st.divider()
    st.markdown('<div class="section-label">Final Heatmap</div>', unsafe_allow_html=True)
    st.image(_render_heatmap(heatmap_acc, scale=scale), channels="RGB", use_container_width=True)


if __name__ == "__main__":
    main()
