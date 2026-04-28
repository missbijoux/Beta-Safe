"""Runtime tuning via environment variables.

Capture
-------
BETASAFE_USE_DXGI (Windows, default on)
    Use DXGI Desktop Duplication via dxcam when available; otherwise mss.

BETASAFE_INACTIVE_REFRESH_EVERY (default 6)
    For monitors where the cursor is *not*, only run full grab+detect every N
    frames; other frames reuse the last mosaic. Set 1 to treat all monitors equally.

Performance / quality
---------------------
BETASAFE_TARGET_FPS (default 12)
BETASAFE_MAX_PROCESS_WIDTH (default 720)
BETASAFE_DETECT_EVERY (default 3)
BETASAFE_IDLE_DIFF (default 1.1)

BETASAFE_MAX_CENSOR_AREA (default 0.92)
    Drop rectangles whose area exceeds this fraction of the whole screen. Helps
    avoid accidental full-screen mosaics when heuristics merge into one giant box.

BETASAFE_OPENCL (default on)
    Use OpenCV OpenCL (UMat) for mosaic/blur when the runtime supports it.

ONNX (optional YOLOv8 export)
----------------------------
BETASAFE_ONNX_PATH
    Path to yolov8*.onnx (Ultralytics export). Empty disables ONNX.

BETASAFE_ONNX_CONF / BETASAFE_ONNX_IOU (defaults 0.35 / 0.45)
    Detector thresholds inside the ONNX head.

BETASAFE_MERGE_IOU (default 0.45)
    NMS IoU when merging ONNX boxes with heuristic boxes.

Adult / sexualized content gating (optional)
-------------------------------------------
Without these, the app mosaics *all* candidate image-like regions (texture heuristic).

BETASAFE_ADULT_HF_MODEL
    Hugging Face model id to run directly via ``transformers`` (no ONNX export).
    Example: ``Falconsai/nsfw_image_detection``.
    This is the simplest setup, but heavier than ONNX.

BETASAFE_ADULT_ONNX_PATH
    Path to a small binary NSFW / adult-content classifier in ONNX format
    (input NCHW RGB float32 ``[0,1]`` is typical). When set and load succeeds,
    only crops scoring above ``BETASAFE_ADULT_ONNX_THRESHOLD`` are mosaiced.

BETASAFE_ADULT_ONNX_THRESHOLD (default 0.72)
BETASAFE_ADULT_ONNX_POS_CLASS (default 1)
    Which softmax index is the 'positive' class for ``(1,2)`` outputs.

BETASAFE_ADULT_SKIN_HEURISTIC (default off)
    Cheap, *very* error-prone skin-tone + texture proxy. Prefer the ONNX classifier.

BETASAFE_ADULT_SKIN_THRESHOLD (default 0.45)

BETASAFE_ADULT_COMBINE (``any`` or ``all``, default ``any``)
    When both ONNX and skin heuristic are enabled, combine scores with OR or AND.

BETASAFE_ADULT_MAX_ONNX_CROPS (default 64)
    Largest-N regions (by area) get ONNX runs first each frame.

BETASAFE_ADULT_TILE (default 96)
    When the adult gate is enabled, also generate a grid of tile candidates
    (smaller crops) and classify those. Smaller tiles catch partial nudity but
    cost more model calls.

BETASAFE_ADULT_MIN_RECTS (default 6)
    If fewer than this many candidate rects are found, tile candidates are added.
"""

from __future__ import annotations

import os
import sys


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(lo, min(hi, v))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return default


TARGET_FPS = _env_int("BETASAFE_TARGET_FPS", 12, lo=1, hi=60)
FRAME_INTERVAL_MS = max(1, int(round(1000.0 / float(TARGET_FPS))))

MAX_PROCESS_WIDTH = _env_int("BETASAFE_MAX_PROCESS_WIDTH", 720, lo=320, hi=4096)
DETECT_EVERY_N_FRAMES = _env_int("BETASAFE_DETECT_EVERY", 3, lo=1, hi=30)
IDLE_DIFF_THRESHOLD = _env_float("BETASAFE_IDLE_DIFF", 1.1)

USE_DXGI = _env_bool("BETASAFE_USE_DXGI", sys.platform == "win32")
INACTIVE_REFRESH_EVERY = _env_int("BETASAFE_INACTIVE_REFRESH_EVERY", 6, lo=1, hi=600)

USE_OPENCL = _env_bool("BETASAFE_OPENCL", True)

ONNX_PATH = os.environ.get("BETASAFE_ONNX_PATH", "").strip()
ONNX_CONF = _env_float("BETASAFE_ONNX_CONF", 0.35)
ONNX_IOU = _env_float("BETASAFE_ONNX_IOU", 0.45)
MERGE_IOU = _env_float("BETASAFE_MERGE_IOU", 0.45)

ADULT_HF_MODEL = os.environ.get("BETASAFE_ADULT_HF_MODEL", "").strip()
ADULT_ONNX_PATH = os.environ.get("BETASAFE_ADULT_ONNX_PATH", "").strip()
ADULT_ONNX_THRESHOLD = _env_float("BETASAFE_ADULT_ONNX_THRESHOLD", 0.72)
ADULT_ONNX_POS_CLASS = _env_int("BETASAFE_ADULT_ONNX_POS_CLASS", 1, lo=0, hi=128)
ADULT_SKIN_HEURISTIC = _env_bool("BETASAFE_ADULT_SKIN_HEURISTIC", False)
ADULT_SKIN_THRESHOLD = _env_float("BETASAFE_ADULT_SKIN_THRESHOLD", 0.45)
ADULT_COMBINE = os.environ.get("BETASAFE_ADULT_COMBINE", "any").strip().lower()
# Despite the name, this also applies to HF direct classifier crops.
# Keep the default low to avoid freezing low-end machines.
ADULT_MAX_ONNX_CROPS = _env_int("BETASAFE_ADULT_MAX_ONNX_CROPS", 8, lo=1, hi=256)
ADULT_TILE = _env_int("BETASAFE_ADULT_TILE", 96, lo=32, hi=256)
ADULT_MIN_RECTS = _env_int("BETASAFE_ADULT_MIN_RECTS", 6, lo=0, hi=128)
MAX_CENSOR_AREA_RATIO = _env_float("BETASAFE_MAX_CENSOR_AREA", 0.92)
