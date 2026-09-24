import os
import json
import time
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import rasterio
from google import genai
from google.genai import types
from ultralytics import YOLO

# ============================================================
# VEGETATION DETECTOR COMPATIBILITY
# ============================================================
# The existing vegetation_detector.py does not currently expose
# `detect_vegetation`, so importing it directly crashes Streamlit.
# Use the project's detector when available; otherwise fall back
# to a local RGB/ExG detector so the application can still run.

try:
    from vegetation_detector import detect_vegetation
except (ImportError, AttributeError):

    def detect_vegetation(image_path):
        """Fallback local RGB vegetation detector.

        Returns the same fields expected by run_vegetation_detection()
        and the Streamlit UI: coverage, regions and result_path.
        """
        import numpy as _np
        from PIL import Image as _Image

        image_path = str(image_path)
        img = _np.asarray(
            _Image.open(image_path).convert("RGB")
        ).astype(_np.float32)

        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # Excess Green Index: vegetation generally has stronger
        # relative green response than red/blue.
        exg = 2.0 * g - r - b

        # Adaptive threshold.
        threshold = float(_np.percentile(exg, 70.0))
        threshold = max(10.0, threshold)
        mask = exg > threshold

        # Reject very dark pixels to reduce false positives.
        brightness = (r + g + b) / 3.0
        mask &= brightness > 25.0

        coverage = float(_np.mean(mask) * 100.0)

        # Estimate vegetation regions using four image quadrants.
        # This avoids adding another dependency just for connected
        # component analysis.
        h, w = mask.shape
        h_mid = max(1, h // 2)
        w_mid = max(1, w // 2)

        quadrants = (
            mask[:h_mid, :w_mid],
            mask[:h_mid, w_mid:],
            mask[h_mid:, :w_mid],
            mask[h_mid:, w_mid:],
        )

        regions = sum(
            1
            for q in quadrants
            if q.size > 0 and float(q.mean()) >= 0.01
        )

        # Create a simple visual vegetation mask for the UI.
        # Vegetation = green, background = original image.
        overlay = img.copy().astype(_np.uint8)
        overlay[mask] = _np.array([40, 220, 80], dtype=_np.uint8)

        result_path = Path(image_path).with_name(
            Path(image_path).stem + "_vegetation_result.png"
        )
        _Image.fromarray(overlay).save(result_path)

        return {
            "coverage": coverage,
            "regions": int(regions),
            "result_path": str(result_path),
            "method": "LOCAL RGB + ExG fallback",
        }

from building_detector import (
    load_model as load_building_model,
    detect_buildings,
    create_building_mask,
    analyze_buildings,
    create_overlay as create_building_overlay,
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NDVI_PATH = BASE_DIR / "ndvi.tif"
VEGETATION_MASK_PATH = BASE_DIR / "vegetation_mask.tif"
DETECTOR_MODEL_PATH = BASE_DIR / "yolo26n-obb.pt"
DETECTOR_CONFIDENCE = 0.15
DETECTOR_IMAGE_SIZE = 1024
BUILDING_MODEL_PATH = BASE_DIR / "models" / "building_model.onnx"
BUILDING_THRESHOLD = 0.4371

# ------------------------------------------------------------
# VAPI / VOICE IMAGE SHARING
# ------------------------------------------------------------
# The FastAPI voice server runs as a separate process, so it cannot
# read Streamlit session_state directly. We therefore keep a copy of
# the currently uploaded RGB image in a stable shared file. The voice
# server reads this same file when Vapi calls an analysis tool.
VOICE_IMAGE_PATH = BASE_DIR / "runs" / "voice_current_image.png"
VOICE_IMAGE_META_PATH = BASE_DIR / "runs" / "voice_current_image.json"

# You can change this in the terminal:
# set GEMINI_MODEL=your-model-name
# NOTE: gemini-2.5-flash is no longer available to new API keys
# (full shutdown Oct 16, 2026). gemini-3.7-flash is the current
# stable Flash model.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

# Used automatically if GEMINI_MODEL is overloaded (503) or rate
# limited (429) after several retries, or otherwise fails. Also used
# if GEMINI_MODEL isn't reachable for any other reason (e.g. a bad
# request or an auth/availability issue for this API key).
# gemini-3.6-flash is the previous-generation stable Flash model and
# Google's own recommended migration target -- unlike gemini-2.5-flash,
# it's still available to new API keys, which is what actually broke
# the old fallback.
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")

# ------------------------------------------------------------
# VAPI WEB VOICE
# ------------------------------------------------------------
# The public Vapi key is safe to use in the browser. Keep private
# Vapi/API keys out of this file. Put VAPI_PUBLIC_KEY in .env.
VAPI_PUBLIC_KEY = os.getenv("VAPI_PUBLIC_KEY", "")
VAPI_ASSISTANT_ID = os.getenv(
    "VAPI_ASSISTANT_ID",
    "83901cfa-0da4-4c13-b1ec-5efefc692b5d",
)


# ============================================================
# GROUND STATION THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    :root{
        --bg:#0A0E14;
        --panel:#10151D;
        --panel-2:#141A23;
        --border:#232B36;
        --border-bright:#3A4552;
        --text:#E8ECF1;
        --text-dim:#93A1B1;
        --amber:#E8A33D;
        --amber-dim:rgba(232,163,61,0.12);
        --green:#4CAF7D;
        --red:#C9614F;
    }

    html, body, [class*="css"]{
        font-family:'Space Grotesk', sans-serif;
    }

    .stApp{
        background-color:var(--bg);
        background-image:
            linear-gradient(rgba(35,43,54,0.35) 1px, transparent 1px),
            linear-gradient(90deg, rgba(35,43,54,0.35) 1px, transparent 1px);
        background-size:44px 44px;
        color:var(--text);
    }

    section[data-testid="stSidebar"]{
        background-color:var(--panel);
        border-right:1px solid var(--border);
    }
    section[data-testid="stSidebar"] *{
        color:var(--text) !important;
    }

    h1,h2,h3{ font-family:'Space Grotesk', sans-serif; }

    p, span, label, div{ color:var(--text); }

    hr, [data-testid="stDivider"]{
        border-color:var(--border) !important;
        opacity:1 !important;
    }

    /* ---------- Hero ---------- */
    .hero-wrap{
        border:1px solid var(--border);
        background:linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
        padding:1.6rem 1.8rem;
        margin-bottom:0.4rem;
        position:relative;
    }
    .hero-wrap::before, .hero-wrap::after{
        content:"";
        position:absolute;
        width:14px; height:14px;
        border-color:var(--amber);
        border-style:solid;
    }
    .hero-wrap::before{
        top:-1px; left:-1px;
        border-width:2px 0 0 2px;
    }
    .hero-wrap::after{
        bottom:-1px; right:-1px;
        border-width:0 2px 2px 0;
    }
    .hero-title{
        font-size:2.1rem;
        font-weight:700;
        letter-spacing:-0.01em;
        margin:0;
        display:flex;
        align-items:center;
        gap:0.6rem;
    }
    .hero-sub{
        color:var(--text-dim);
        font-size:0.95rem;
        margin:0.35rem 0 1rem 0;
    }
    .status-strip{
        display:flex;
        flex-wrap:wrap;
        gap:0.6rem;
        font-family:'IBM Plex Mono', monospace;
        font-size:0.74rem;
        letter-spacing:0.02em;
    }
    .status-chip{
        display:flex; align-items:center; gap:0.45rem;
        border:1px solid var(--border-bright);
        background:var(--bg);
        padding:0.3rem 0.65rem;
        color:var(--text-dim);
    }
    .status-dot{
        width:7px; height:7px; border-radius:50%;
        background:var(--border-bright);
        flex-shrink:0;
    }
    .status-dot.on{ background:var(--green); box-shadow:0 0 6px rgba(76,175,125,0.7); }
    .status-dot.off{ background:var(--red); }
    .status-chip b{ color:var(--text); font-weight:600; }

    /* ---------- Section / panel headers ---------- */
    .panel-head{
        display:flex; align-items:baseline; gap:0.7rem;
        border-bottom:1px solid var(--border);
        padding:0.3rem 0 0.7rem 0;
        margin:1.8rem 0 1rem 0;
    }
    .panel-head .step-tag{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.7rem;
        color:var(--amber);
        border:1px solid var(--border-bright);
        padding:2px 8px;
        letter-spacing:0.03em;
        white-space:nowrap;
    }
    .panel-head h2{
        font-size:1.25rem;
        font-weight:600;
        margin:0;
        color:var(--text);
    }

    /* ---------- Generic panel ---------- */
    .console-panel{
        border:1px solid var(--border);
        background:var(--panel);
        padding:1.1rem 1.2rem;
    }
    .console-row{
        display:flex; justify-content:space-between;
        font-family:'IBM Plex Mono', monospace;
        font-size:0.82rem;
        padding:0.35rem 0;
        border-bottom:1px solid var(--border);
        color:var(--text-dim);
    }
    .console-row:last-child{ border-bottom:none; }
    .console-row b{ color:var(--text); font-weight:600; }
    .console-panel .panel-label{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.72rem;
        color:var(--text-dim);
        letter-spacing:0.03em;
        margin-bottom:0.6rem;
    }

    /* ---------- Directional / compass readout ---------- */
    .compass-grid{
        display:grid;
        grid-template-columns:repeat(4, 1fr);
        gap:1px;
        background:var(--border);
        border:1px solid var(--border);
        margin-top:1px;
    }
    .compass-cell{
        background:var(--panel);
        padding:0.85rem 1rem;
    }
    .compass-cell .compass-label{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.7rem;
        color:var(--text-dim);
        letter-spacing:0.03em;
        margin-bottom:0.35rem;
    }
    .compass-cell .compass-value{
        font-family:'IBM Plex Mono', monospace;
        font-size:1.15rem;
        font-weight:600;
        color:var(--green);
    }
    .compass-cell .compass-sub{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.68rem;
        color:var(--text-dim);
        margin-top:0.2rem;
    }
    .compass-cell.highest{
        box-shadow:inset 0 0 0 1px var(--amber);
    }
    .compass-cell.highest .compass-value{
        color:var(--amber);
    }
    .compass-cell.highest .compass-label::after{
        content:" · PEAK";
        color:var(--amber);
    }
    @media (max-width:900px){
        .compass-grid{ grid-template-columns:repeat(2,1fr); }
    }

    /* ---------- Stat grid (NDVI) ---------- */
    .stat-grid{
        display:grid;
        grid-template-columns:repeat(4, 1fr);
        gap:1px;
        background:var(--border);
        border:1px solid var(--border);
    }
    .stat-cell{
        background:var(--panel);
        padding:1rem 1.1rem;
    }
    .stat-cell .stat-label{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.7rem;
        color:var(--text-dim);
        letter-spacing:0.03em;
        margin-bottom:0.4rem;
    }
    .stat-cell .stat-value{
        font-family:'IBM Plex Mono', monospace;
        font-size:1.6rem;
        font-weight:600;
        color:var(--amber);
    }
    @media (max-width:900px){
        .stat-grid{ grid-template-columns:repeat(2,1fr); }
    }

    /* ---------- Buttons ---------- */
    .stButton>button{
        background:var(--panel) !important;
        color:var(--text) !important;
        border:1px solid var(--border-bright) !important;
        border-radius:0 !important;
        font-family:'IBM Plex Mono', monospace !important;
        font-size:0.78rem !important;
        padding:0.5rem 0.8rem !important;
        transition:border-color 0.15s ease, color 0.15s ease;
    }
    .stButton>button:hover{
        border-color:var(--amber) !important;
        color:var(--amber) !important;
    }
    .stButton>button:active{
        background:var(--amber-dim) !important;
    }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploaderDropzone"]{
        background:var(--panel) !important;
        border:1px dashed var(--border-bright) !important;
        border-radius:0 !important;
    }

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"]{
        background:var(--panel) !important;
        border:1px solid var(--border) !important;
        border-radius:0 !important;
    }
    [data-testid="stChatInput"] textarea{
        background:var(--panel) !important;
        border:1px solid var(--border-bright) !important;
        border-radius:0 !important;
        color:var(--text) !important;
        font-family:'Space Grotesk', sans-serif !important;
    }

    /* ---------- Alerts ---------- */
    [data-testid="stAlert"]{
        border-radius:0 !important;
        border:1px solid var(--border-bright) !important;
        font-family:'IBM Plex Mono', monospace !important;
        font-size:0.82rem !important;
    }

    /* ---------- Expander ---------- */
    [data-testid="stExpander"]{
        border:1px solid var(--border) !important;
        border-radius:0 !important;
        background:var(--panel) !important;
    }

    /* ---------- Analyzing indicator ---------- */
    .analyzing-line{
        display:flex;
        align-items:center;
        gap:0.6rem;
        font-family:'IBM Plex Mono', monospace;
        font-size:0.85rem;
        color:var(--amber);
    }
    .analyzing-dots span{
        display:inline-block;
        width:5px; height:5px;
        border-radius:50%;
        background:var(--amber);
        margin-left:3px;
        animation:blink 1.2s infinite ease-in-out;
    }
    .analyzing-dots span:nth-child(2){ animation-delay:0.2s; }
    .analyzing-dots span:nth-child(3){ animation-delay:0.4s; }
    @keyframes blink{
        0%, 80%, 100% { opacity:0.15; }
        40% { opacity:1; }
    }
    .cursor-blink{
        display:inline-block;
        width:7px;
        background:var(--amber);
        margin-left:2px;
        animation:cursorblink 0.9s infinite;
    }
    @keyframes cursorblink{
        0%, 50% { opacity:1; }
        51%, 100% { opacity:0; }
    }

    /* ---------- Feedback buttons ---------- */
    div[data-testid="column"] .stButton>button{
        padding:0.25rem 0.55rem !important;
        font-size:0.78rem !important;
    }
    .feedback-note{
        font-family:'IBM Plex Mono', monospace;
        font-size:0.72rem;
        color:var(--text-dim);
        padding-top:0.3rem;
    }

    /* ---------- UI UPDATE: layout / tabs / downloads ---------- */
    /* keep the last content clear of the pinned chat bar */
    .block-container, [data-testid="stMainBlockContainer"]{
        padding-bottom:9rem !important;
    }
    .stat-grid.auto{
        grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));
    }
    .stTabs [data-baseweb="tab-list"]{
        gap:0.3rem;
        border-bottom:1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"]{
        background:transparent;
        border-radius:0;
        padding:0.5rem 0.9rem;
        font-family:'IBM Plex Mono', monospace;
        font-size:0.8rem;
        color:var(--text-dim);
    }
    .stTabs [data-baseweb="tab"] p{
        color:inherit !important;
        font-family:inherit !important;
        font-size:inherit !important;
    }
    .stTabs [aria-selected="true"],
    .stTabs [aria-selected="true"] p{
        color:var(--amber) !important;
    }
    .stTabs [data-baseweb="tab-highlight"]{
        background-color:var(--amber) !important;
    }
    .stDownloadButton>button{
        background:var(--panel) !important;
        color:var(--text) !important;
        border:1px solid var(--border-bright) !important;
        border-radius:0 !important;
        font-family:'IBM Plex Mono', monospace !important;
        font-size:0.78rem !important;
    }
    .stDownloadButton>button:hover{
        border-color:var(--amber) !important;
        color:var(--amber) !important;
    }

    footer, #MainMenu{ visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def html_block(markup):
    """
    Render raw HTML through st.markdown safely.

    Streamlit's Markdown parser follows CommonMark: any line indented
    4+ spaces (with no preceding HTML-block opener on that same line)
    is treated as a literal code block, not HTML. Our f-strings are
    written with Python-style indentation, so nested/joined HTML (like
    the compass grid, built from a list of separately-indented cell
    strings) was landing over that threshold and printing as raw tags
    instead of rendering. Stripping leading whitespace from each line
    and joining without newlines sidesteps the ambiguity entirely.
    """
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def panel_head(step_tag, title):
    html_block(
        f"""
        <div class="panel-head">
            <span class="step-tag">{step_tag}</span>
            <h2>{title}</h2>
        </div>
        """
    )


# ============================================================
# GEMINI
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

# Gemini is optional for fast local OBJECT/GIS/HYBRID answers.
# Only VISION questions need the API.
client = None
gemini_ready = False
gemini_init_error = None

if api_key:
    try:
        client = genai.Client(api_key=api_key)
        gemini_ready = True
    except Exception as e:
        gemini_init_error = str(e)
else:
    gemini_init_error = "GEMINI_API_KEY is not set."


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_key" not in st.session_state:
    st.session_state.image_key = None

if "image" not in st.session_state:
    st.session_state.image = None

if "image_for_model" not in st.session_state:
    st.session_state.image_for_model = None

if "ndvi_stats" not in st.session_state:
    st.session_state.ndvi_stats = None

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

if "detections" not in st.session_state:
    st.session_state.detections = None

if "detection_image" not in st.session_state:
    st.session_state.detection_image = None

if "detection_error" not in st.session_state:
    st.session_state.detection_error = None

if "building_detections" not in st.session_state:
    st.session_state.building_detections = None

if "building_detection_image" not in st.session_state:
    st.session_state.building_detection_image = None

if "building_coverage" not in st.session_state:
    st.session_state.building_coverage = 0.0

if "building_error" not in st.session_state:
    st.session_state.building_error = None

if "vegetation_result" not in st.session_state:
    st.session_state.vegetation_result = None

if "vegetation_error" not in st.session_state:
    st.session_state.vegetation_error = None

if "ndvi_requested" not in st.session_state:
    st.session_state.ndvi_requested = False

if "voice_image_sync_error" not in st.session_state:
    st.session_state.voice_image_sync_error = None


# ============================================================
# NDVI
# ============================================================

@st.cache_data
def calculate_ndvi_statistics(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        with rasterio.open(path) as src:
            ndvi = src.read(1).astype("float32")
            nodata = src.nodata

        valid = np.isfinite(ndvi)

        if nodata is not None:
            valid &= ndvi != nodata

        values = ndvi[valid]

        if values.size == 0:
            return None

        total = int(values.size)

        # ------------------------------------------------------
        # Directional breakdown.
        # Rasterio reads north-up rasters top-to-bottom, so row 0
        # is the northern edge and row -1 is the southern edge;
        # column 0 is the western edge and column -1 is the eastern
        # edge. We split the raster into North/South halves (by row)
        # and East/West halves (by column) and score vegetation
        # separately in each, so we can tell the model roughly where
        # the vegetation in the scene is concentrated.
        # ------------------------------------------------------

        height, width = ndvi.shape

        def region_stats(mask_slice):
            region_valid = valid & mask_slice
            region_values = ndvi[region_valid]

            if region_values.size == 0:
                return None

            return {
                "mean_ndvi": float(np.mean(region_values)),
                "vegetation_percentage": float(
                    np.sum(region_values >= 0.30) / region_values.size * 100
                ),
            }

        row_mask = np.zeros((height, width), dtype=bool)
        row_mask[: height // 2, :] = True  # North half

        col_mask = np.zeros((height, width), dtype=bool)
        col_mask[:, : width // 2] = True  # West half

        regions = {
            "North": region_stats(row_mask),
            "South": region_stats(~row_mask),
            "East": region_stats(~col_mask),
            "West": region_stats(col_mask),
        }

        scored_regions = {
            name: stats["vegetation_percentage"]
            for name, stats in regions.items()
            if stats is not None
        }

        highest_region = (
            max(scored_regions, key=scored_regions.get)
            if scored_regions
            else None
        )

        return {
            "mean_ndvi": float(np.mean(values)),
            "min_ndvi": float(np.min(values)),
            "max_ndvi": float(np.max(values)),
            "vegetation_percentage": float(
                np.sum(values >= 0.30) / total * 100
            ),
            "moderate_percentage": float(
                np.sum(values >= 0.40) / total * 100
            ),
            "dense_percentage": float(
                np.sum(values >= 0.50) / total * 100
            ),
            "total_pixels": total,
            "regions": regions,
            "highest_vegetation_region": highest_region,
        }

    except Exception as e:
        st.warning(f"Could not read ndvi.tif: {e}")
        return None


def load_vegetation_mask():
    if not VEGETATION_MASK_PATH.exists():
        return None

    try:
        with rasterio.open(VEGETATION_MASK_PATH) as src:
            return src.read(1)
    except Exception:
        return None


# ============================================================
# IMAGE LOADING
# ============================================================

def load_uploaded_image(uploaded_file):
    """
    Supports PNG/JPG/JPEG and common TIFF files.
    For TIFF files that PIL cannot read directly, rasterio
    attempts to read the first three bands as RGB.
    """

    uploaded_file.seek(0)

    try:
        image = Image.open(uploaded_file).convert("RGB")
        return image
    except Exception:
        pass

    uploaded_file.seek(0)

    try:
        with rasterio.MemoryFile(uploaded_file.read()) as memfile:
            with memfile.open() as src:
                if src.count >= 3:
                    arr = src.read([1, 2, 3])
                    arr = np.moveaxis(arr, 0, -1)

                    arr = arr.astype("float32")
                    low = np.nanpercentile(arr, 2)
                    high = np.nanpercentile(arr, 98)

                    if high > low:
                        arr = (arr - low) / (high - low)

                    arr = np.clip(arr, 0, 1)
                    arr = (arr * 255).astype("uint8")

                    return Image.fromarray(arr)

                if src.count == 1:
                    arr = src.read(1).astype("float32")

                    low = np.nanpercentile(arr, 2)
                    high = np.nanpercentile(arr, 98)

                    if high > low:
                        arr = (arr - low) / (high - low)

                    arr = np.clip(arr, 0, 1)
                    arr = (arr * 255).astype("uint8")

                    return Image.fromarray(arr).convert("RGB")

    except Exception as e:
        raise ValueError(f"Unsupported or corrupted image: {e}")

    raise ValueError("Could not read the uploaded image.")


# ============================================================
# MODEL-SIZED IMAGE
# ============================================================

def prepare_image_for_model(img, max_dim=1280):
    """
    Vision models don't get more accurate above ~1280px on the long
    edge, but a full-resolution satellite image (often 4000px+) takes
    much longer to upload and process. Downscale a copy for Gemini
    while keeping the original at full resolution for on-screen preview.
    """
    width, height = img.size

    if max(width, height) <= max_dim:
        return img

    scale = max_dim / float(max(width, height))
    new_size = (int(width * scale), int(height * scale))

    return img.resize(new_size, Image.LANCZOS)


def sync_image_for_voice(image, original_name, original_size):
    """
    Save the currently uploaded image to the shared path used by the
    FastAPI/Vapi voice backend. This keeps voice analysis synchronized
    with the image currently visible in Streamlit.
    """
    try:
        VOICE_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Save the RGB representation used by the local detectors.
        image.convert("RGB").save(VOICE_IMAGE_PATH, format="PNG")

        # Small metadata file is useful for debugging/status checks.
        metadata = {
            "original_name": str(original_name),
            "original_size": int(original_size),
            "shared_image": str(VOICE_IMAGE_PATH),
            "width": int(image.width),
            "height": int(image.height),
        }
        VOICE_IMAGE_META_PATH.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        return None
    except Exception as e:
        return str(e)


# ============================================================
# BUILDING DETECTION — HOTOSM
# ============================================================

@st.cache_resource
def load_building_detector():
    """Load the HOTOSM building segmentation model once."""
    if not BUILDING_MODEL_PATH.exists():
        return None
    return load_building_model()


def run_building_detection(img):
    """Run HOTOSM building segmentation on the uploaded RGB image."""
    try:
        session = load_building_detector()
        if session is None:
            return None, [], 0.0, f"Building model not found: {BUILDING_MODEL_PATH}"

        # building_detector.py expects an OpenCV BGR image.
        image_rgb = np.array(img.convert("RGB"))
        image_bgr = image_rgb[:, :, ::-1].copy()

        probability_map = detect_buildings(
            session,
            image_bgr,
        )

        mask = create_building_mask(
            probability_map
        )

        buildings = analyze_buildings(
            mask
        )

        coverage = (
            float(np.count_nonzero(mask))
            / float(mask.size)
            * 100.0
        )

        overlay_bgr = create_building_overlay(
            image_bgr,
            mask,
            buildings,
        )

        # Streamlit expects RGB arrays.
        overlay_rgb = overlay_bgr[:, :, ::-1].copy()

        return overlay_rgb, buildings, coverage, None

    except Exception as e:
        return None, [], 0.0, str(e)


# ============================================================
# VEGETATION DETECTION — LOCAL RGB
# ============================================================

def run_vegetation_detection(img):
    """Run the local RGB vegetation detector on the uploaded image."""
    try:
        output_dir = BASE_DIR / "runs"
        output_dir.mkdir(exist_ok=True)
        input_path = output_dir / "current_uploaded_rgb.png"

        # The vegetation detector works on RGB/JPG/PNG pixels.
        # Save the already-normalized uploaded image rather than using
        # the unrelated fixed ndvi.tif raster.
        img.convert("RGB").save(input_path, format="PNG")

        result = detect_vegetation(str(input_path))
        return result, None
    except Exception as e:
        return None, f"Vegetation detector error: {e}"


# ============================================================
# SATELLITE OBJECT DETECTION — DOTA OBB
# ============================================================

@st.cache_resource
def load_satellite_detector():
    """Load the pretrained DOTA oriented-bounding-box detector once."""
    if not DETECTOR_MODEL_PATH.exists():
        return None
    return YOLO(str(DETECTOR_MODEL_PATH))


def run_satellite_detection(img):
    """
    Run DOTA OBB detection on the uploaded image.

    Returns:
        annotated_image: PIL image with oriented boxes drawn
        detections: list of dictionaries with class/confidence
        error: None or a human-readable error message
    """
    model = load_satellite_detector()

    if model is None:
        return None, [], (
            f"Detector model not found: {DETECTOR_MODEL_PATH.name}. "
            "Place yolo26n-obb.pt beside app.py."
        )

    try:
        # Keep the uploaded image in RGB and pass the PIL image
        # directly to Ultralytics.
        image = img.convert("RGB")

        results = model.predict(
            source=image,
            conf=DETECTOR_CONFIDENCE,
            iou=0.7,
            imgsz=DETECTOR_IMAGE_SIZE,
            max_det=100,
            verbose=False,
        )

        result = results[0]
        detections = []

        if result.obb is not None:
            class_ids = result.obb.cls.cpu().tolist()
            confidences = result.obb.conf.cpu().tolist()

            for class_id, confidence in zip(class_ids, confidences):
                class_id = int(class_id)
                detections.append(
                    {
                        "class": result.names[class_id],
                        "confidence": float(confidence),
                    }
                )

        # Ultralytics can render OBB predictions directly.
        annotated_image = result.plot(pil=True, conf=True, labels=True)

        return annotated_image, detections, None

    except Exception as e:
        return None, [], f"Satellite detector error: {e}"


def summarize_detections(detections):
    """Create a compact class-count summary for the UI and Gemini."""
    counts = {}
    for item in detections:
        name = item["class"]
        counts[name] = counts.get(name, 0) + 1

    return counts


def make_detection_context(detections):
    """Turn detector output into evidence Gemini can safely use."""
    if not detections:
        return (
            "DOTA OBJECT DETECTOR RESULT\n"
            "No objects were detected above the configured confidence "
            f"threshold ({DETECTOR_CONFIDENCE:.2f})."
        )

    counts = summarize_detections(detections)
    count_lines = "\n".join(
        f"- {name}: {count}"
        for name, count in sorted(counts.items())
    )

    confidence_lines = "\n".join(
        f"- {item['class']}: {item['confidence']:.2f}"
        for item in detections
    )

    return f"""
DOTA OBJECT DETECTOR RESULT

Total detected objects: {len(detections)}

OBJECT COUNTS:
{count_lines}

INDIVIDUAL DETECTIONS:
{confidence_lines}

These are model detections, not human-verified ground truth.
Use them as the authoritative object-count evidence for questions
about detected object classes/counts. Do not invent additional
objects or counts.
"""


# ============================================================
# GIS CONTEXT
# ============================================================

def make_gis_context(stats):
    if not stats:
        return "No QGIS NDVI raster is currently available."

    regions = stats.get("regions") or {}
    highest_region = stats.get("highest_vegetation_region")

    region_lines = []
    for name in ("North", "South", "East", "West"):
        region = regions.get(name)
        if region:
            region_lines.append(
                f"{name}: mean NDVI {region['mean_ndvi']:.4f}, "
                f"vegetation coverage {region['vegetation_percentage']:.2f}%"
            )
        else:
            region_lines.append(f"{name}: no valid data")

    region_block = "\n".join(region_lines)
    highest_line = (
        f"Region with highest vegetation coverage: {highest_region}"
        if highest_region
        else "Region with highest vegetation coverage: not available"
    )

    return f"""
QGIS NDVI REFERENCE DATA

Mean NDVI: {stats['mean_ndvi']:.4f}
Minimum NDVI: {stats['min_ndvi']:.4f}
Maximum NDVI: {stats['max_ndvi']:.4f}

NDVI >= 0.30: {stats['vegetation_percentage']:.2f}%
NDVI >= 0.40: {stats['moderate_percentage']:.2f}%
NDVI >= 0.50: {stats['dense_percentage']:.2f}%

Valid raster pixels: {stats['total_pixels']}

DIRECTIONAL VEGETATION DISTRIBUTION
(North/South split by row, East/West split by column — these are
independent halves of the same raster, not four separate quadrants)

{region_block}

{highest_line}
"""



# ============================================================
# QUERY ROUTER
# ============================================================

def route_query(question):
    """Route a question to the smallest useful local evidence pipeline."""
    q = question.lower().strip()

    building_terms = (
        "building", "buildings", "house", "houses",
        "structure", "structures", "building count",
        "number of buildings", "how many buildings",
    )

    object_terms = (
        "object", "objects", "plane", "planes", "aircraft", "vehicle",
        "vehicles", "ship", "ships", "tank", "tanks", "harbor", "bridge",
        "helicopter", "helicopters", "airport", "tennis court",
        "basketball court", "baseball diamond", "roundabout",
        "storage tank", "track field", "soccer field", "swimming pool",
        "how many", "count", "detected", "detection", "detections",
    )

    specific_object_terms = (
        "object", "objects", "plane", "planes", "aircraft", "vehicle",
        "vehicles", "ship", "ships", "tank", "tanks", "harbor", "bridge",
        "helicopter", "helicopters", "airport", "tennis court",
        "basketball court", "baseball diamond", "roundabout",
        "storage tank", "track field", "soccer field", "swimming pool",
    )

    vegetation_terms = (
        "vegetation", "greenery", "plant", "plants", "tree", "trees",
        "crop", "crops", "forest", "forests", "vegetated",
        "vegetation cover", "vegetation coverage", "healthy vegetation",
        "dense vegetation", "moderate vegetation", "green area",
    )

    ndvi_terms = (
        "ndvi", "qgis", "raster", "pixel", "red band", "nir",
        "near infrared", "near-infrared",
    )

    building_hit = any(term in q for term in building_terms)
    object_hit = any(term in q for term in object_terms)
    specific_object_hit = any(term in q for term in specific_object_terms)
    vegetation_hit = any(term in q for term in vegetation_terms)
    ndvi_hit = any(term in q for term in ndvi_terms)

    # Multiple local evidence sources = HYBRID. No Gemini is required
    # unless the question is a genuine visual reasoning question.
    if building_hit and (specific_object_hit or vegetation_hit or ndvi_hit):
        tools = ["HOTOSM BUILDING DETECTOR"]
        if specific_object_hit:
            tools.append("DOTA OBJECT DETECTOR")
        if vegetation_hit:
            tools.append("LOCAL VEGETATION DETECTOR")
        if ndvi_hit:
            tools.append("QGIS / NDVI")
        return {
            "route": "HYBRID",
            "tools": tools,
            "reason": "The question combines building detection with another local analysis source.",
        }

    if object_hit and (vegetation_hit or ndvi_hit):
        tools = ["DOTA OBJECT DETECTOR"]
        if vegetation_hit:
            tools.append("LOCAL VEGETATION DETECTOR")
        if ndvi_hit:
            tools.append("QGIS / NDVI")
        return {
            "route": "HYBRID",
            "tools": tools,
            "reason": "The question combines object detection with vegetation or GIS evidence.",
        }

    if vegetation_hit and ndvi_hit:
        return {
            "route": "HYBRID",
            "tools": ["LOCAL VEGETATION DETECTOR", "QGIS / NDVI"],
            "reason": "The question asks for both RGB vegetation evidence and NDVI/GIS evidence.",
        }

    if building_hit:
        return {
            "route": "BUILDING",
            "tools": ["HOTOSM BUILDING DETECTOR"],
            "reason": "The question is primarily about building detection or building count.",
        }

    if object_hit:
        return {
            "route": "OBJECT",
            "tools": ["DOTA OBJECT DETECTOR"],
            "reason": "The question is primarily about detected objects or object counts.",
        }

    if vegetation_hit:
        return {
            "route": "VEGETATION",
            "tools": ["LOCAL VEGETATION DETECTOR"],
            "reason": "The question is primarily about visible vegetation in the uploaded RGB image.",
        }

    if ndvi_hit:
        return {
            "route": "GIS",
            "tools": ["QGIS / NDVI"],
            "reason": "The question can be answered directly from the connected NDVI/GIS statistics.",
        }

    return {
        "route": "VISION",
        "tools": ["GEMINI VISION"],
        "reason": "The question is a general visual/scene-understanding question.",
    }


def local_fast_answer(route_info, question, stats, detections, building_detections=None, building_coverage=0.0, vegetation_result=None):
    """Answer questions that have sufficient local detector/GIS evidence."""
    route = route_info["route"]
    q = question.lower().strip()

    def object_answer():
        counts = summarize_detections(detections or [])
        if not counts:
            return (
                f"🛰️ **No objects were detected above the detector "
                f"confidence threshold ({DETECTOR_CONFIDENCE:.2f}).**"
            )

        aliases = {
            "plane": ["plane"], "planes": ["plane"], "aircraft": ["plane"],
            "vehicle": ["large vehicle", "small vehicle"],
            "vehicles": ["large vehicle", "small vehicle"],
            "ship": ["ship"], "ships": ["ship"],
            "tank": ["storage tank"], "tanks": ["storage tank"],
            "helicopter": ["helicopter"], "helicopters": ["helicopter"],
            "bridge": ["bridge"], "harbor": ["harbor"],
            "airport": ["plane"], "tennis court": ["tennis court"],
            "basketball court": ["basketball court"],
            "baseball diamond": ["baseball diamond"], "roundabout": ["roundabout"],
            "storage tank": ["storage tank"], "track field": ["ground track field"],
            "soccer field": ["soccer ball field"], "swimming pool": ["swimming pool"],
        }

        requested = []
        for phrase, classes in aliases.items():
            if phrase in q:
                for cls in classes:
                    if cls not in requested:
                        requested.append(cls)

        if requested:
            lines = [f"- **{cls.title()}**: {counts.get(cls, 0)}" for cls in requested]
            return "🛩️ **Object detection result**\n\n" + "\n".join(lines)

        lines = [f"- **{name.title()}**: {count}" for name, count in sorted(counts.items())]
        return f"🛰️ **{len(detections)} objects detected**\n\n" + "\n".join(lines)

    def building_answer():
        buildings = building_detections or []
        if building_detections is None:
            return "🏠 **Building detector data is not available.**"
        return (
            f"🏠 **{len(buildings)} building regions detected.**\n\n"
            f"Estimated building coverage: **{building_coverage:.1f}%**.\n\n"
            f"Model threshold: **{BUILDING_THRESHOLD:.4f}**."
        )

    def vegetation_answer():
        if vegetation_result is None:
            return "🌱 **Vegetation detector data is not available.**"

        coverage = float(vegetation_result.get("coverage", 0.0))
        regions = int(vegetation_result.get("regions", 0))

        if coverage <= 0:
            return (
                "🌱 **No vegetation regions were detected by the local RGB detector.**\n\n"
                "This is an RGB-based estimate, not NDVI."
            )

        return (
            f"🌱 **Visible vegetation detected.**\n\n"
            f"Estimated RGB vegetation coverage: **{coverage:.1f}%**.\n\n"
            f"Detected vegetation regions: **{regions}**.\n\n"
            "Gemini was **not used** for this vegetation result. "
            "This is not NDVI."
        )

    def gis_answer():
        if not stats:
            return "🌱 **NDVI data is not available.** The connected `ndvi.tif` could not be read."
        mean = stats["mean_ndvi"]
        minimum = stats["min_ndvi"]
        maximum = stats["max_ndvi"]
        veg = stats["vegetation_percentage"]
        moderate = stats["moderate_percentage"]
        dense = stats["dense_percentage"]
        highest = stats.get("highest_vegetation_region")

        if "where" in q or "concentrated" in q:
            if highest:
                region = stats["regions"].get(highest)
                return (
                    f"🌱 **Vegetation is most concentrated in the {highest} half** "
                    f"of the connected NDVI raster, with about {region['vegetation_percentage']:.1f}% "
                    f"of valid pixels at NDVI ≥ 0.30."
                )
            return "🌱 A highest-vegetation region could not be determined from the NDVI raster."

        if "is there vegetation" in q or q.startswith("is there"):
            return (
                f"🌱 **Yes.** About **{veg:.1f}%** of valid raster pixels have "
                f"NDVI ≥ 0.30."
                if veg > 0
                else "🌱 **No pixels meet the NDVI ≥ 0.30 vegetation threshold.**"
            )

        if "how much vegetation" in q or "vegetation percentage" in q or "vegetation coverage" in q:
            return (
                f"🌱 **NDVI vegetation coverage: {veg:.1f}%** (NDVI ≥ 0.30).\n\n"
                f"Moderate threshold (≥ 0.40): **{moderate:.1f}%**\n\n"
                f"Dense threshold (≥ 0.50): **{dense:.1f}%**"
            )

        if "min" in q and "max" in q:
            return (
                f"🌱 **NDVI range:** {minimum:.3f} to {maximum:.3f}\n\n"
                f"Mean NDVI: **{mean:.3f}**"
            )

        return (
            f"🌱 **Mean NDVI: {mean:.3f}**\n\n"
            f"Range: **{minimum:.3f} → {maximum:.3f}**\n\n"
            f"NDVI ≥ 0.30: **{veg:.1f}%** of valid pixels."
        )

    if route == "BUILDING":
        return building_answer()
    if route == "OBJECT":
        return object_answer()
    if route == "VEGETATION":
        return vegetation_answer()
    if route == "GIS":
        return gis_answer()

    if route == "HYBRID":
        parts = []
        tools = route_info.get("tools", [])
        if "HOTOSM BUILDING DETECTOR" in tools:
            parts.append(building_answer())
        if "DOTA OBJECT DETECTOR" in tools:
            parts.append(object_answer())
        if "LOCAL VEGETATION DETECTOR" in tools:
            parts.append(vegetation_answer())
        if "QGIS / NDVI" in tools:
            parts.append(gis_answer())
        return "\n\n---\n\n".join(parts) if parts else None

    return None


def make_route_context(route_info):
    """Create routing instructions that keep Gemini grounded in selected evidence."""
    tools = ", ".join(route_info["tools"])
    return f"""
QUERY ROUTER RESULT

Route: {route_info['route']}
Selected tools: {tools}
Reason: {route_info['reason']}

Use the selected evidence sources above. The router is a routing aid,
not evidence itself. For exact object counts, use the DOTA detector result.
For building counts, use the HOTOSM building detector result.
For visible vegetation questions, use the local RGB vegetation detector result.
For NDVI/GIS measurements, use the supplied QGIS raster statistics.
For visual interpretation, inspect the uploaded image.
"""


# ============================================================
# GEMINI QUERY
# ============================================================

GEMINI_CONFIG = types.GenerateContentConfig(
    # "low" keeps Gemini 3.x from spending extra time on deep internal
    # reasoning it doesn't need for a descriptive vision Q&A task. This
    # is the single biggest lever on response time for this app.
    thinking_config=types.ThinkingConfig(thinking_level="low"),
)


def build_prompt(question, stats, previous_messages, detections=None, route_info=None, building_detections=None, building_coverage=0.0):
    gis_context = make_gis_context(stats)
    detection_context = make_detection_context(detections or [])
    building_context = (
        f"Detected building regions: {len(building_detections or [])}. "
        f"Estimated building coverage: {building_coverage:.1f}%."
        if building_detections is not None
        else "Building detector data is not available."
    )
    route_info = route_info or route_query(question)
    route_context = make_route_context(route_info)

    history = ""

    # Keep only the latest few turns so prompts do not grow forever.
    for item in previous_messages[-6:]:
        role = item.get("role", "user")
        text = item.get("content", "")
        history += f"{role.upper()}: {text}\n"

    return f"""
You are SatQuery AI, an intelligent assistant specialized in:

- satellite imagery
- remote sensing
- GIS
- NDVI
- land-use interpretation
- urban analysis
- environmental analysis

The user has uploaded a satellite image.

Answer the user's question using the actual uploaded image.

USER QUESTION:
{question}

PREVIOUS CONVERSATION:
{history}

GIS REFERENCE:
{gis_context}

OBJECT DETECTION REFERENCE:
{detection_context}

BUILDING DETECTION REFERENCE:
{building_context}

{route_context}

RULES:

1. For visual questions, inspect the uploaded image.
2. For NDVI/GIS questions, use the supplied GIS values.
3. Clearly distinguish visual observations from GIS measurements.
4. Never invent coordinates.
5. Never invent exact object counts.
6. Do not claim an exact percentage from RGB appearance alone.
7. If the image does not provide enough evidence, say so.
8. If the GIS raster is separate from the uploaded image, do not pretend
   that you calculated NDVI directly from the RGB image.
9. Answer the user's exact question first.
10. Keep the response concise but informative.
11. Use simple language suitable for a non-GIS expert.
12. For object-count questions, use the DOTA detector result rather than guessing from the image.
13. For building-count questions, use the HOTOSM building detector result rather than guessing from the image.
14. If the detector found no objects, say that the detector found none above its confidence threshold; do not turn that into a claim that the scene contains no objects at all.
14. If useful, structure the answer with short bullets.
15. The directional (North/South/East/West) vegetation figures are
    coarse halves of the raster, not precise zones — treat them as
    an approximate distribution, not exact boundaries.

You are an AI satellite-image assistant, not a generic chatbot.
"""


def _is_temporary_error(error_text):
    upper = error_text.upper()
    return (
        "503" in error_text
        or "UNAVAILABLE" in upper
        or "429" in error_text
        or "RESOURCE_EXHAUSTED" in upper
    )


def stream_gemini_answer(image_for_model, question, stats, previous_messages, detections=None, route_info=None, building_detections=None, building_coverage=0.0):
    """
    Yields text chunks as Gemini generates them, so the answer appears
    progressively in the UI instead of all at once after a long wait.

    Resilience strategy:
    1. Try GEMINI_MODEL, retrying transient errors (503/429) with
       exponential backoff + jitter, up to `retries_per_model` times.
    2. If the primary model is still unavailable, fall back to
       GEMINI_FALLBACK_MODEL and repeat the same retry logic there.
    3. Only surface an error to the user if both models fail.

    Each model's final error is tracked separately (`model_errors`)
    so that if every model fails, the user sees exactly what went
    wrong with each one instead of only the last error encountered.
    """
    route_info = route_info or route_query(question)
    prompt = build_prompt(
        question,
        stats,
        previous_messages,
        detections=detections,
        route_info=route_info,
    )

    models_to_try = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        models_to_try.append(GEMINI_FALLBACK_MODEL)

    retries_per_model = 4
    model_errors = {}

    for model_index, model_name in enumerate(models_to_try):
        is_fallback = model_index > 0
        last_error_for_model = None

        if is_fallback:
            prev_model = models_to_try[model_index - 1]
            prev_error = model_errors.get(prev_model, "unknown error")
            yield (
                f"_({prev_model} failed — {prev_error}. "
                f"Retrying with {model_name})_\n\n"
            )

        for attempt in range(retries_per_model):
            try:
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=[image_for_model, prompt],
                    config=GEMINI_CONFIG,
                )

                received_any = False

                for chunk in stream:
                    if chunk.text:
                        received_any = True
                        yield chunk.text

                if received_any:
                    return

                last_error_for_model = "Gemini returned an empty response."
                break  # try next model rather than retrying an empty reply

            except Exception as e:
                error = str(e)
                last_error_for_model = error

                if _is_temporary_error(error) and attempt < retries_per_model - 1:
                    wait = min(2 ** attempt, 15) + random.uniform(0, 1)
                    time.sleep(wait)
                    continue

                break  # exhausted retries (or non-temporary error) -> next model

        model_errors[model_name] = last_error_for_model

    error_lines = "\n".join(
        f"- {name}: {err}" for name, err in model_errors.items()
    )

    yield (
        "SatQuery could not complete the AI analysis.\n\n"
        "All configured models are currently unavailable:\n\n"
        f"{error_lines}\n\n"
        "If this mentions a 404 or 'no longer available', the model name "
        "itself needs updating in GEMINI_MODEL / GEMINI_FALLBACK_MODEL. "
        "If it mentions 503/UNAVAILABLE or 429/RESOURCE_EXHAUSTED, this is "
        "usually temporary high demand on Google's side — try again shortly."
    )


# ============================================================
# HERO / HEADER
# ============================================================

ndvi_connected = NDVI_PATH.exists()
mask_connected = VEGETATION_MASK_PATH.exists()
detector_connected = DETECTOR_MODEL_PATH.exists()

html_block(
    f"""
    <div class="hero-wrap">
        <div class="hero-title">🛰️ SatQuery AI</div>
        <div class="hero-sub">Visual + GIS satellite analysis console — Gemini vision layered on your QGIS NDVI data.</div>
        <div class="status-strip">
            <div class="status-chip"><span class="status-dot on"></span>MODEL&nbsp;<b>{GEMINI_MODEL}</b></div>
            <div class="status-chip"><span class="status-dot {'on' if ndvi_connected else 'off'}"></span>NDVI LINK&nbsp;<b>{'CONNECTED' if ndvi_connected else 'OFFLINE'}</b></div>
            <div class="status-chip"><span class="status-dot {'on' if mask_connected else 'off'}"></span>MASK LINK&nbsp;<b>{'CONNECTED' if mask_connected else 'OFFLINE'}</b></div>
<div class="status-chip"><span class="status-dot {'on' if detector_connected else 'off'}"></span>OBJECT DETECTOR&nbsp;<b>{'READY' if detector_connected else 'OFFLINE'}</b></div>
        </div>
    </div>
    """
)


# ============================================================
# VAPI WEB VOICE WIDGET (INLINE, TAP-TO-TALK)
# ============================================================
#
# Why a custom component instead of components.html():
# components.html() renders inside an iframe whose origin is "null".
# Vapi's audio engine (Daily) calls postMessage() with the page's
# origin and crashes with "Invalid target origin 'null'". A Streamlit
# custom component is served from a real URL (http://localhost:8501/...)
# so it has a proper origin, and it stays inline on the same page.
#
# The component's index.html is written automatically to
# ./voice_component/index.html on startup -- nothing to copy by hand.

VOICE_COMPONENT_DIR = BASE_DIR / "voice_component"

VOICE_COMPONENT_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root{
    --bg:#0A0E14; --panel:#10151D; --border:#232B36; --border-bright:#3A4552;
    --text:#E8ECF1; --dim:#7C8A9A; --amber:#E8A33D; --green:#4CAF7D; --red:#C9614F;
  }
  *{ box-sizing:border-box; }
  html, body{ margin:0; background:transparent; color:var(--text);
    font-family:'IBM Plex Mono', monospace; }
  .wrap{
    display:flex; flex-direction:column; align-items:center; gap:0.7rem;
    padding:0.6rem 0 0.2rem 0;
  }
  .row{ display:flex; align-items:center; gap:1rem; }

  #mic{
    width:84px; height:84px; border-radius:50%;
    background:var(--panel); color:var(--amber);
    border:2px solid var(--border-bright);
    font-size:2rem; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    transition:border-color .15s ease, background .15s ease, transform .1s ease;
  }
  #mic:hover{ border-color:var(--amber); }
  #mic:active{ transform:scale(0.96); }
  #mic.connecting{ border-color:var(--amber); animation:pulse 1s infinite; }
  #mic.live{
    background:rgba(76,175,125,0.12); border-color:var(--green); color:var(--green);
    animation:ring 1.6s infinite;
  }
  #mic.speaking{
    background:rgba(232,163,61,0.14); border-color:var(--amber); color:var(--amber);
    animation:ring-amber 1.2s infinite;
  }
  #mic.error{ border-color:var(--red); color:var(--red); }

  #mute{
    width:44px; height:44px; border-radius:50%;
    background:var(--panel); color:var(--text);
    border:1px solid var(--border-bright); font-size:1.05rem; cursor:pointer;
    display:none;
  }
  #mute.show{ display:block; }
  #mute.muted{ border-color:var(--red); color:var(--red); }

  @keyframes pulse{ 0%,100%{opacity:1} 50%{opacity:.45} }
  @keyframes ring{
    0%{ box-shadow:0 0 0 0 rgba(76,175,125,.55); }
    100%{ box-shadow:0 0 0 22px rgba(76,175,125,0); }
  }
  @keyframes ring-amber{
    0%{ box-shadow:0 0 0 0 rgba(232,163,61,.55); }
    100%{ box-shadow:0 0 0 22px rgba(232,163,61,0); }
  }

  #status{ font-size:.82rem; color:var(--dim); letter-spacing:.02em;
    text-align:center; max-width:95%; word-break:break-word; }
  #status b{ color:var(--text); }

  #vol{ width:180px; height:4px; background:var(--border); overflow:hidden; }
  #vol > div{ height:100%; width:0%; background:var(--green); transition:width .08s linear; }

  #log{
    width:100%; max-height:200px; overflow-y:auto;
    border:1px solid var(--border); background:var(--panel);
    padding:.5rem .7rem; font-size:.78rem; line-height:1.55; color:var(--dim);
    display:none;
  }
  #log > div{ padding:.3rem .6rem; margin:.25rem 0; border-left:2px solid var(--border-bright); }
  #log .u{ color:var(--text); border-left-color:var(--text); }
  #log .a{ color:var(--amber); border-left-color:var(--amber); }
  #log .live{ opacity:.55; font-style:italic; }
  @keyframes think{ 0%,100%{opacity:1} 50%{opacity:.4} }
  #mic.thinking{
    background:rgba(232,163,61,0.10); border-color:var(--amber); color:var(--amber);
    animation:think 1s infinite;
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="row">
    <button id="mic" title="Tap to talk">🎙️</button>
    <button id="mute" title="Mute / unmute your microphone">🔊</button>
  </div>
  <div id="vol"><div></div></div>
  <div id="status">Tap the mic to <b>start talking</b> with SatQuery AI</div>
  <div id="log"></div>
</div>

<script type="module">
  // ---------- Streamlit component protocol (minimal) ----------
  function sendToStreamlit(type, data){
    window.parent.postMessage(
      Object.assign({ isStreamlitMessage: true, type: type }, data || {}), "*"
    );
  }

  let PUBLIC_KEY = "";
  let ASSISTANT_ID = "";

  window.addEventListener("message", (event) => {
    const msg = event.data;
    if (msg && msg.type === "streamlit:render"){
      const args = msg.args || {};
      PUBLIC_KEY = args.public_key || "";
      ASSISTANT_ID = args.assistant_id || "";
    }
  });

  sendToStreamlit("streamlit:componentReady", { apiVersion: 1 });
  const H_IDLE = 230, H_LOG = 440;
  function setFrameHeight(h){ sendToStreamlit("streamlit:setFrameHeight", { height: h }); }
  setFrameHeight(H_IDLE);

  // ---------- UI ----------
  const micBtn = document.getElementById('mic');
  const muteBtn = document.getElementById('mute');
  const statusEl = document.getElementById('status');
  const logEl = document.getElementById('log');
  const volBar = document.querySelector('#vol > div');

  let vapi = null;
  let inCall = false;
  let starting = false;
  let muted = false;

  function setStatus(html){ statusEl.innerHTML = html; }
  function setMicState(cls){ micBtn.className = cls || ''; }

  let liveEl = null;

  function showLog(){
    if (logEl.style.display !== 'block'){
      logEl.style.display = 'block';
      setFrameHeight(H_LOG);
    }
  }

  function clearLive(){
    if (liveEl){ liveEl.remove(); liveEl = null; }
  }

  function addLine(role, text, live){
    showLog();
    const prefix = (role === 'user' ? 'You: ' : 'SatQuery: ');
    if (live){
      if (!liveEl){ liveEl = document.createElement('div'); logEl.appendChild(liveEl); }
      liveEl.className = (role === 'user' ? 'u' : 'a') + ' live';
      liveEl.textContent = prefix + text;
    } else {
      clearLive();
      const div = document.createElement('div');
      div.className = role === 'user' ? 'u' : 'a';
      div.textContent = prefix + text;
      logEl.appendChild(div);
    }
    logEl.scrollTop = logEl.scrollHeight;
  }

  function resetUI(){
    inCall = false;
    starting = false;
    muted = false;
    setMicState('');
    micBtn.textContent = '🎙️';
    muteBtn.classList.remove('show', 'muted');
    muteBtn.textContent = '🔊';
    volBar.style.width = '0%';
  }

  function errText(e){
    if (!e) return 'unknown error';
    if (typeof e === 'string') return e;
    return e.errorMsg || (e.error && e.error.message) || e.message || JSON.stringify(e);
  }

  async function loadVapi(){
    if (vapi) return vapi;

    // Try two CDNs in case one is blocked on the user's network.
    const sources = [
      'https://esm.sh/@vapi-ai/web',
      'https://cdn.jsdelivr.net/npm/@vapi-ai/web/+esm',
    ];
    let mod = null, lastErr = null;
    for (const src of sources){
      try { mod = await import(src); break; }
      catch (e){ lastErr = e; }
    }
    if (!mod) throw new Error('Could not load the Vapi SDK (' + lastErr + ')');

    let Vapi = mod.default || mod.Vapi;
    if (Vapi && Vapi.default) Vapi = Vapi.default;  // handle CJS interop
    vapi = new Vapi(PUBLIC_KEY);

    vapi.on('call-start', () => {
      inCall = true; starting = false;
      setMicState('live');
      micBtn.textContent = '⏹';
      muteBtn.classList.add('show');
      setStatus('<b style="color:var(--green)">LISTENING</b> — go ahead and talk. Tap ⏹ to end.');
    });

    vapi.on('call-end', () => {
      resetUI();
      setStatus('Call ended. Tap the mic to <b>talk again</b>.');
    });

    vapi.on('speech-start', () => {
      setMicState('speaking');
      setStatus('<b style="color:var(--amber)">SATQUERY IS SPEAKING</b>');
    });

    vapi.on('speech-end', () => {
      if (inCall){
        setMicState('live');
        setStatus('<b style="color:var(--green)">LISTENING</b> — go ahead and talk. Tap ⏹ to end.');
      }
    });

    vapi.on('volume-level', (v) => {
      volBar.style.width = Math.min(100, Math.round(v * 100)) + '%';
    });

    vapi.on('message', (m) => {
      if (!m) return;

      // live + final transcript lines
      if (m.type === 'transcript' && m.transcript){
        addLine(m.role, m.transcript, m.transcriptType !== 'final');
        return;
      }

      // the assistant is running the image-analysis tools
      if (m.type === 'tool-calls' || m.type === 'function-call'){
        setMicState('thinking');
        setStatus('<b style="color:var(--amber)">ANALYZING IMAGE</b> — running the specialist tools…');
        return;
      }

      if (m.type === 'tool-calls-result' && inCall){
        setMicState('live');
      }
    });

    vapi.on('error', (e) => {
      console.error('Vapi error', e);
      resetUI();
      setMicState('error');
      setStatus('<b style="color:var(--red)">Voice error:</b> ' + errText(e));
    });

    return vapi;
  }

  async function startCall(){
    if (starting || inCall) return;

    if (!PUBLIC_KEY || !ASSISTANT_ID){
      setMicState('error');
      setStatus('<b style="color:var(--red)">Voice not configured yet.</b> Reload the page and try again.');
      return;
    }

    starting = true;
    setMicState('connecting');
    setStatus('Connecting… <b>allow the microphone</b> if your browser asks.');

    // Ask for mic permission up front so failures give a clear message.
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(t => t.stop());
    } catch (e){
      resetUI();
      setMicState('error');
      setStatus(
        '<b style="color:var(--red)">Microphone blocked.</b> Click the 🔒 icon in the ' +
        'address bar → allow Microphone for this site, then tap the mic again.'
      );
      return;
    }

    try {
      const v = await loadVapi();
      await v.start(ASSISTANT_ID);
    } catch (e){
      resetUI();
      setMicState('error');
      setStatus('<b style="color:var(--red)">Could not start:</b> ' + errText(e));
    }
  }

  function endCall(){
    try { if (vapi) vapi.stop(); } catch (e){ console.error(e); }
    resetUI();
    setStatus('Call ended. Tap the mic to <b>talk again</b>.');
  }

  micBtn.addEventListener('click', () => {
    if (inCall || starting){ endCall(); } else { startCall(); }
  });

  muteBtn.addEventListener('click', () => {
    if (!vapi || !inCall) return;
    muted = !muted;
    vapi.setMuted(muted);
    muteBtn.classList.toggle('muted', muted);
    muteBtn.textContent = muted ? '🔇' : '🔊';
    setStatus(muted
      ? '<b style="color:var(--red)">MIC MUTED</b> — tap 🔇 to unmute.'
      : '<b style="color:var(--green)">LISTENING</b> — go ahead and talk. Tap ⏹ to end.');
  });
</script>
</body>
</html>
"""


def _ensure_voice_component_files():
    """Write voice_component/index.html beside app.py (only if changed)."""
    VOICE_COMPONENT_DIR.mkdir(exist_ok=True)
    index_path = VOICE_COMPONENT_DIR / "index.html"
    try:
        current = (
            index_path.read_text(encoding="utf-8")
            if index_path.exists()
            else None
        )
        if current != VOICE_COMPONENT_HTML:
            index_path.write_text(VOICE_COMPONENT_HTML, encoding="utf-8")
    except Exception as e:
        st.warning(f"Could not write voice component files: {e}")


_ensure_voice_component_files()

_vapi_voice_component = components.declare_component(
    "satquery_vapi_voice",
    path=str(VOICE_COMPONENT_DIR),
)


def render_vapi_voice_widget():
    """
    Inline tap-to-talk voice assistant (stays in the SAME tab).

    - Tap the mic  -> starts the call (browser asks for mic permission once)
    - Tap again    -> ends the call
    - Mute button  -> mutes/unmutes your mic during the call
    - Live transcript of the conversation is shown under the button
    """
    if not VAPI_PUBLIC_KEY:
        st.warning(
            "🎙️ Voice assistant is not configured yet. Add "
            "`VAPI_PUBLIC_KEY=...` to your `.env` file and restart SatQuery."
        )
        return

    _vapi_voice_component(
        public_key=VAPI_PUBLIC_KEY,
        assistant_id=VAPI_ASSISTANT_ID,
        key="satquery_vapi_voice",
        default=None,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🛰️ SatQuery AI")
    st.caption("Upload one satellite image and ask multiple questions about the same scene.")

    html_block(
        """
        <div class="console-panel" style="margin-top:0.8rem;">
            <div class="panel-label">PIPELINE</div>
            <div class="console-row"><span>01</span><b>Upload satellite image</b></div>
            <div class="console-row"><span>02</span><b>DOTA object detection</b></div>
            <div class="console-row"><span>03</span><b>Gemini vision analysis</b></div>
            <div class="console-row"><span>04</span><b>QGIS NDVI analysis</b></div>
            <div class="console-row"><span>05</span><b>Query routing</b></div>
            <div class="console-row"><span>06</span><b>Evidence-based answer</b></div>
        </div>
        """
    )

    html_block(
        f"""
        <div class="console-panel" style="margin-top:0.9rem;">
            <div class="panel-label">GIS LAYERS</div>
            <div class="console-row"><span>ndvi.tif</span><b style="color:{'var(--green)' if ndvi_connected else 'var(--red)'}">{'CONNECTED' if ndvi_connected else 'NOT FOUND'}</b></div>
            <div class="console-row"><span>vegetation_mask.tif</span><b style="color:{'var(--green)' if mask_connected else 'var(--text-dim)'}">{'CONNECTED' if mask_connected else 'NOT FOUND'}</b></div>
        </div>
        """
    )

    st.write("")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# STEP 1 — UPLOAD
# ============================================================

panel_head("STEP 01", "Upload satellite image")

uploaded_file = st.file_uploader(
    "Choose a satellite image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    help="Upload RGB satellite imagery or a readable GeoTIFF.",
    label_visibility="collapsed",
)


# ============================================================
# AFTER UPLOAD
# ============================================================

if uploaded_file is None:
    st.info("Upload a satellite image above to start using SatQuery AI.")

    panel_head("REF", "Example questions")

    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown('<div class="console-panel">🌱&nbsp; Is there vegetation?</div>', unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="console-panel">🏙️&nbsp; What objects are visible?</div>', unsafe_allow_html=True)
    with q3:
        st.markdown('<div class="console-panel">🗺️&nbsp; What is the NDVI?</div>', unsafe_allow_html=True)

    st.stop()


# Detect a new uploaded image.
image_key = f"{uploaded_file.name}_{uploaded_file.size}"

if st.session_state.image_key != image_key:
    try:
        image = load_uploaded_image(uploaded_file)

        st.session_state.image = image
        st.session_state.image_for_model = prepare_image_for_model(image)
        st.session_state.image_key = image_key
        st.session_state.messages = []
        st.session_state.ndvi_stats = calculate_ndvi_statistics(
            str(NDVI_PATH)
        )

        # ----------------------------------------------------
        # SYNC CURRENT IMAGE TO THE VOICE BACKEND
        # ----------------------------------------------------
        # FastAPI/Vapi runs outside Streamlit, so give it a stable
        # shared file containing the same RGB image the user just
        # uploaded.
        st.session_state.voice_image_sync_error = sync_image_for_voice(
            image,
            uploaded_file.name,
            uploaded_file.size,
        )

        # Do not run any expensive specialist detector just because an image
        # was uploaded. Detectors are triggered lazily when the user asks a
        # question that actually needs them.
        st.session_state.detections = None
        st.session_state.detection_image = None
        st.session_state.detection_error = None

        st.session_state.building_detections = None
        st.session_state.building_detection_image = None
        st.session_state.building_coverage = 0.0
        st.session_state.building_error = None
        st.session_state.vegetation_result = None
        st.session_state.vegetation_error = None
        st.session_state.ndvi_requested = False

    except Exception as e:
        st.error(f"Could not load image: {e}")
        st.stop()

image = st.session_state.image
image_for_model = st.session_state.image_for_model
ndvi_stats = st.session_state.ndvi_stats


# ============================================================
# STEP 2 — IMAGE PREVIEW
# ============================================================

panel_head("STEP 02", "Uploaded satellite image")

left, right = st.columns([2.2, 1])

with left:
    st.image(image, caption=uploaded_file.name, use_container_width=True)

with right:
    html_block(
        f"""
        <div class="console-panel" style="margin-bottom:0.8rem;">
            <div class="panel-label">IMAGE STATUS</div>
            <div class="console-row"><span>File</span><b>{uploaded_file.name}</b></div>
            <div class="console-row"><span>Dimensions</span><b>{image.width} × {image.height}</b></div>
            <div class="console-row"><span>Vision model</span><b style="color:var(--green)">READY</b></div>
            <div class="console-row"><span>Voice image</span><b style="color:{'var(--green)' if VOICE_IMAGE_PATH.exists() and not st.session_state.voice_image_sync_error else 'var(--red)'}">{'SYNCED' if VOICE_IMAGE_PATH.exists() and not st.session_state.voice_image_sync_error else 'OFFLINE'}</b></div>
        </div>
        """
    )

    if st.session_state.voice_image_sync_error:
        st.warning(
            "Voice image sync failed: "
            + st.session_state.voice_image_sync_error
        )

    # UI: live analysis summary lives here (filled at the end of the script)
    summary_slot = st.empty()


# ============================================================
# STEP 03 — REAL-TIME VOICE ASSISTANT
# ============================================================

panel_head("STEP 03", "Talk to SatQuery AI")

st.markdown(
    "Ask naturally about the uploaded image. Tap the mic to start talking — "
    "the voice assistant uses the same synchronized image as the specialist "
    "analysis tools.",
)
render_vapi_voice_widget()


# ============================================================
# ANALYSIS RESULTS  (UI: one tabbed panel, redrawn at end of each run)
# ============================================================

def _png_bytes(img_like):
    """Encode a PIL image / RGB array / image path as PNG bytes (for download)."""
    import io

    try:
        if isinstance(img_like, (str, Path)):
            return Path(img_like).read_bytes()
        if isinstance(img_like, np.ndarray):
            img_like = Image.fromarray(img_like.astype("uint8"))
        buf = io.BytesIO()
        img_like.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _download_button(img_like, filename, key):
    data = _png_bytes(img_like)
    if data:
        st.download_button(
            "⬇ Download annotated image",
            data,
            file_name=filename,
            mime="image/png",
            key=key,
            use_container_width=True,
        )


def _rows_panel(label, rows):
    body = "".join(
        f'<div class="console-row"><span>{k}</span><b>{v}</b></div>'
        for k, v in rows
    )
    html_block(
        f'<div class="console-panel" style="margin-bottom:0.8rem;">'
        f'<div class="panel-label">{label}</div>{body}</div>'
    )


def render_summary_panel():
    """Live summary shown beside the uploaded image."""
    det = st.session_state.detections
    bld = st.session_state.building_detections
    veg = st.session_state.vegetation_result
    ndvi_on = bool(st.session_state.ndvi_requested and ndvi_stats)

    def cell(text, done):
        color = "var(--green)" if done else "var(--text-dim)"
        return f'<span style="color:{color}">{text}</span>'

    rows = [
        (
            "Objects",
            cell(
                f"{len(det)} found" if det is not None else "not run",
                det is not None,
            ),
        ),
        (
            "Buildings",
            cell(
                f"{len(bld)} regions · {st.session_state.building_coverage:.1f}%"
                if bld is not None
                else "not run",
                bld is not None,
            ),
        ),
        (
            "Vegetation (RGB)",
            cell(
                f"{veg.get('coverage', 0.0):.1f}%" if veg else "not run",
                bool(veg),
            ),
        ),
        (
            "NDVI mean",
            cell(
                f"{ndvi_stats['mean_ndvi']:.3f}" if ndvi_on else "not run",
                ndvi_on,
            ),
        ),
    ]

    if det:
        counts = summarize_detections(det)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        rows.append(("Top classes", ", ".join(f"{n} ×{c}" for n, c in top)))

    _rows_panel("ANALYSIS SUMMARY", rows)

    if det is None and bld is None and not veg and not ndvi_on:
        st.caption("Nothing analysed yet — ask a question below or use the mic.")


def render_results_panel():
    """One tabbed panel with every analysis result produced so far."""
    det = st.session_state.detections
    det_img = st.session_state.detection_image
    det_err = st.session_state.detection_error
    bld = st.session_state.building_detections
    bld_img = st.session_state.building_detection_image
    bld_err = st.session_state.building_error
    veg = st.session_state.vegetation_result
    veg_err = st.session_state.vegetation_error

    spec = []
    if det is not None or det_err:
        spec.append(("🛩️ Objects", "obj"))
    if bld_img is not None or bld_err:
        spec.append(("🏠 Buildings", "bld"))
    if veg is not None or veg_err:
        spec.append(("🌱 Vegetation", "veg"))
    if st.session_state.ndvi_requested:
        spec.append(("🗺️ NDVI", "ndvi"))

    if not spec:
        return

    # Put the tab for the most recently used tool first.
    focus = st.session_state.get("results_focus")
    spec.sort(key=lambda item: 0 if item[1] == focus else 1)

    panel_head("RESULTS", "Analysis results")
    tabs = st.tabs([label for label, _ in spec])

    for tab, (_, key) in zip(tabs, spec):
        with tab:

            # ---------------- OBJECTS ----------------
            if key == "obj":
                if det_err:
                    st.warning(det_err)
                elif det_img is not None:
                    left_col, right_col = st.columns([2.2, 1])
                    with left_col:
                        st.image(
                            det_img,
                            caption="DOTA OBB detector output",
                            use_container_width=True,
                        )
                    with right_col:
                        counts = summarize_detections(det or [])
                        _rows_panel(
                            "DETECTION STATUS",
                            [
                                ("Total objects", len(det or [])),
                                ("Model", "YOLO26n-OBB"),
                                ("Threshold", f"{DETECTOR_CONFIDENCE:.2f}"),
                            ],
                        )
                        if counts:
                            _rows_panel(
                                "DETECTED CLASSES",
                                [(n, c) for n, c in sorted(counts.items())],
                            )
                        else:
                            st.caption(
                                "No objects detected above the confidence threshold."
                            )
                        _download_button(det_img, "satquery_objects.png", "dl_objects")

            # ---------------- BUILDINGS ----------------
            elif key == "bld":
                if bld_err:
                    st.warning(bld_err)
                elif bld_img is not None:
                    left_col, right_col = st.columns([2.2, 1])
                    with left_col:
                        st.image(
                            bld_img,
                            caption="HOTOSM building detection output",
                            use_container_width=True,
                        )
                    with right_col:
                        _rows_panel(
                            "BUILDING DETECTION STATUS",
                            [
                                ("Building regions", len(bld or [])),
                                ("Coverage", f"{st.session_state.building_coverage:.1f}%"),
                                ("Model", "HOTOSM DINOv3-S"),
                                ("Threshold", f"{BUILDING_THRESHOLD:.4f}"),
                            ],
                        )
                        st.caption(
                            "Building regions come from the local segmentation "
                            "model, not from Gemini guesses."
                        )
                        _download_button(bld_img, "satquery_buildings.png", "dl_buildings")

            # ---------------- VEGETATION ----------------
            elif key == "veg":
                if veg_err:
                    st.warning(veg_err)
                elif veg:
                    left_col, right_col = st.columns([2.2, 1])
                    result_path = veg.get("result_path")
                    with left_col:
                        if result_path and Path(result_path).exists():
                            st.image(
                                result_path,
                                caption="Local RGB vegetation detection output",
                                use_container_width=True,
                            )
                    with right_col:
                        _rows_panel(
                            "VEGETATION DETECTION STATUS",
                            [
                                ("Coverage", f"{veg.get('coverage', 0.0):.1f}%"),
                                ("Regions", veg.get("regions", 0)),
                                ("Method", "LOCAL RGB"),
                                ("Gemini", "NOT USED"),
                            ],
                        )
                        st.caption(
                            "Estimated locally from the RGB image. This is not NDVI."
                        )
                        if result_path and Path(result_path).exists():
                            _download_button(result_path, "satquery_vegetation.png", "dl_vegetation")

            # ---------------- NDVI ----------------
            elif key == "ndvi":
                if ndvi_stats:
                    html_block(
                        f"""
                        <div class="stat-grid">
                            <div class="stat-cell">
                                <div class="stat-label">MEAN NDVI</div>
                                <div class="stat-value">{ndvi_stats['mean_ndvi']:.3f}</div>
                            </div>
                            <div class="stat-cell">
                                <div class="stat-label">NDVI ≥ 0.30</div>
                                <div class="stat-value">{ndvi_stats['vegetation_percentage']:.1f}%</div>
                            </div>
                            <div class="stat-cell">
                                <div class="stat-label">NDVI ≥ 0.40</div>
                                <div class="stat-value">{ndvi_stats['moderate_percentage']:.1f}%</div>
                            </div>
                            <div class="stat-cell">
                                <div class="stat-label">NDVI ≥ 0.50</div>
                                <div class="stat-value">{ndvi_stats['dense_percentage']:.1f}%</div>
                            </div>
                        </div>
                        """
                    )
                    st.caption(
                        "These measurements come from the connected QGIS ndvi.tif "
                        "raster, not from RGB colour estimation."
                    )

                    regions = ndvi_stats.get("regions") or {}
                    highest_region = ndvi_stats.get("highest_vegetation_region")

                    compass_cells = []
                    for name in ("North", "South", "East", "West"):
                        region = regions.get(name)
                        cell_class = (
                            "compass-cell highest"
                            if name == highest_region
                            else "compass-cell"
                        )
                        if region:
                            value_html = f"{region['vegetation_percentage']:.1f}%"
                            sub_html = f"mean {region['mean_ndvi']:.3f}"
                        else:
                            value_html = "—"
                            sub_html = "no data"

                        compass_cells.append(
                            f"""
                            <div class="{cell_class}">
                                <div class="compass-label">{name.upper()}</div>
                                <div class="compass-value">{value_html}</div>
                                <div class="compass-sub">{sub_html}</div>
                            </div>
                            """
                        )

                    html_block(
                        f"""
                        <div class="panel-label" style="margin:1.1rem 0 0.4rem 0;">DIRECTIONAL VEGETATION DISTRIBUTION</div>
                        <div class="compass-grid">
                            {''.join(compass_cells)}
                        </div>
                        """
                    )

                    if highest_region:
                        st.caption(
                            f"Vegetation coverage is highest in the {highest_region} half of the raster."
                        )
                else:
                    st.info(
                        "No ndvi.tif was found beside app.py. "
                        "Visual AI analysis will still work."
                    )


# The panel is drawn into this slot at the END of the script so it always
# shows the latest detector output from the current run.
results_slot = st.empty()


# ============================================================
# STEP 5 — CHAT
# ============================================================

panel_head("STEP 04", "Ask SatQuery AI")

st.caption("Ask multiple questions about the same uploaded image. SatQuery automatically routes each question to the relevant analysis tools.")

example_questions = [
    "What objects were detected?",
    "How many buildings are there?",
    "Is there vegetation?",
    "What is the NDVI of this area?",
    "Describe this scene.",
]

cols = st.columns(len(example_questions))

for i, example in enumerate(example_questions):
    with cols[i]:
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_question = example


pending = st.session_state.pop("pending_question", "")

question = st.chat_input("Ask SatQuery about this satellite image...")

if pending:
    question = pending


# ============================================================
# DISPLAY OLD CHAT
# ============================================================

def render_feedback_row(msg_index):
    current = st.session_state.feedback.get(msg_index)

    up_col, down_col, note_col = st.columns([0.08, 0.08, 0.84])

    with up_col:
        if st.button("👍", key=f"fb_up_{msg_index}"):
            st.session_state.feedback[msg_index] = "up"
            st.rerun()

    with down_col:
        if st.button("👎", key=f"fb_down_{msg_index}"):
            st.session_state.feedback[msg_index] = "down"
            st.rerun()

    with note_col:
        if current == "up":
            st.markdown('<div class="feedback-note">Marked helpful ✓</div>', unsafe_allow_html=True)
        elif current == "down":
            st.markdown('<div class="feedback-note">Marked not helpful — noted</div>', unsafe_allow_html=True)


for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_feedback_row(i)


# ============================================================
# NEW QUESTION
# ============================================================

if question:
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    # Decide which analysis pipeline should handle this question.
    route_info = route_query(question)

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    route_label = {
        "BUILDING": "🏠 BUILDING DETECTOR · LOCAL",
        "OBJECT": "🛩️ OBJECT DETECTOR · LOCAL",
        "GIS": "🗺️ QGIS / NDVI · LOCAL",
        "VEGETATION": "🌱 VEGETATION DETECTOR · LOCAL",
        "HYBRID": "🛰️ HYBRID · LOCAL",
        "VISION": "🧠 GEMINI VISION",
    }.get(route_info["route"], route_info["route"])

    st.caption(f"QUERY ROUTER → {route_label}  ·  {route_info['reason']}")

    with st.chat_message("assistant"):
        # ------------------------------------------------------------
        # LAZY DETECTION
        # ------------------------------------------------------------
        # GIS/NDVI output is also on-demand.
        if "QGIS / NDVI" in route_info.get("tools", []):
            st.session_state.ndvi_requested = True

        # Specialist models run ONLY when the user's question needs them.
        # Uploading an image by itself never triggers building/object
        # detection.
        if route_info["route"] in ("BUILDING", "HYBRID", "OBJECT"):

            # Building detector: run only when this question asks for
            # buildings/structures and the route includes the building tool.
            if "HOTOSM BUILDING DETECTOR" in route_info.get("tools", []):
                if st.session_state.building_detections is None:
                    if not BUILDING_MODEL_PATH.exists():
                        st.session_state.building_error = (
                            f"Building model not found: {BUILDING_MODEL_PATH.name}"
                        )
                    else:
                        with st.spinner("🏠 Analyzing buildings in the uploaded image..."):
                            (
                                building_detection_image,
                                building_detections,
                                building_coverage,
                                building_error,
                            ) = run_building_detection(image)

                        st.session_state.building_detection_image = building_detection_image
                        st.session_state.building_detections = building_detections
                        st.session_state.building_coverage = building_coverage
                        st.session_state.building_error = building_error

            # DOTA detector: run only when this question needs object detection.
            if "DOTA OBJECT DETECTOR" in route_info.get("tools", []):
                if st.session_state.detections is None:
                    if not DETECTOR_MODEL_PATH.exists():
                        st.session_state.detection_error = (
                            f"Detector model not found: {DETECTOR_MODEL_PATH.name}"
                        )
                    else:
                        with st.spinner("🛰️ Detecting objects in the uploaded image..."):
                            (
                                detection_image,
                                detections,
                                detection_error,
                            ) = run_satellite_detection(image)

                        st.session_state.detection_image = detection_image
                        st.session_state.detections = detections
                        st.session_state.detection_error = detection_error

        # Local RGB vegetation detector: run ONLY when the question needs it.
        if "LOCAL VEGETATION DETECTOR" in route_info.get("tools", []):
            if st.session_state.vegetation_result is None and st.session_state.vegetation_error is None:
                with st.spinner("🌱 Analyzing vegetation in the uploaded image..."):
                    vegetation_result, vegetation_error = run_vegetation_detection(image)
                st.session_state.vegetation_result = vegetation_result
                st.session_state.vegetation_error = vegetation_error

        # Fast path: use data that has now been computed locally.
        # No detector runs unless the question requested that evidence.
        answer = local_fast_answer(
            route_info=route_info,
            question=question,
            stats=ndvi_stats,
            detections=st.session_state.detections,
            building_detections=st.session_state.building_detections,
            building_coverage=st.session_state.building_coverage,
            vegetation_result=st.session_state.vegetation_result,
        )

        if answer is not None:
            st.markdown(answer)
        else:
            # Only genuine visual/scene-understanding questions reach Gemini.
            placeholder = st.empty()

            if not gemini_ready:
                answer = (
                    "🧠 **Gemini Vision is not available right now.**\n\n"
                    "Add `GEMINI_API_KEY=YOUR_API_KEY` to the `.env` file "
                    "beside `app.py`, then restart Streamlit."
                )
                if gemini_init_error and gemini_init_error != "GEMINI_API_KEY is not set.":
                    answer += f"\n\nInitialization error: `{gemini_init_error}`"
                placeholder.markdown(answer)
            else:
                placeholder.markdown(
                    '<div class="analyzing-line">🛰️ Analyzing satellite image'
                    '<span class="analyzing-dots"><span></span><span></span><span></span></span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                full_response = ""

                for chunk in stream_gemini_answer(
                    image_for_model=image_for_model,
                    question=question,
                    stats=ndvi_stats,
                    previous_messages=st.session_state.messages[:-1],
                    detections=st.session_state.detections,
                    route_info=route_info,
                    building_detections=st.session_state.building_detections,
                    building_coverage=st.session_state.building_coverage,
                ):
                    full_response += chunk
                    placeholder.markdown(
                        full_response + '<span class="cursor-blink">&nbsp;</span>',
                        unsafe_allow_html=True,
                    )

                placeholder.markdown(full_response)
                answer = full_response

        # Annotated evidence now lives in the tabbed "Analysis results" panel
        # (drawn at the end of the script), so it is not repeated here.
        _tab_for_tool = {
            "DOTA OBJECT DETECTOR": "obj",
            "HOTOSM BUILDING DETECTOR": "bld",
            "LOCAL VEGETATION DETECTOR": "veg",
            "QGIS / NDVI": "ndvi",
        }
        _used_tabs = [
            _tab_for_tool[t]
            for t in route_info.get("tools", [])
            if t in _tab_for_tool
        ]
        if _used_tabs:
            st.session_state.results_focus = _used_tabs[0]
            st.caption(
                "📊 Annotated results are in the **Analysis results** panel above."
            )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        render_feedback_row(len(st.session_state.messages) - 1)


# ============================================================
# DRAW LIVE RESULT PANELS  (UI)
# ============================================================

with results_slot.container():
    render_results_panel()

with summary_slot.container():
    render_summary_panel()


# ============================================================
# VEGETATION MASK
# ============================================================

if VEGETATION_MASK_PATH.exists():
    with st.expander("View QGIS vegetation mask"):
        mask = load_vegetation_mask()

        if mask is not None:
            mask_display = (
                np.isfinite(mask) & (mask > 0)
            ).astype(np.uint8) * 255

            mask_image = Image.fromarray(mask_display)

            st.image(
                mask_image,
                caption="Vegetation mask generated from the QGIS raster.",
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

html_block(
    """
    <div style="margin-top:2rem; padding-top:1rem; border-top:1px solid var(--border);
                font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--text-dim);">
        SATQUERY AI · PROTOTYPE BUILD · DOTA OBB + GEMINI VISION + QGIS + NDVI
    </div>
    """
)
