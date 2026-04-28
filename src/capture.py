"""Cross-platform screen capture via mss (BGR uint8)."""

from __future__ import annotations

from typing import Any

import mss
import numpy as np


def grab_region_bgr(sct: Any, left: int, top: int, width: int, height: int) -> np.ndarray:
    """Grab a virtual-desktop rectangle. *sct* is an open mss instance (reuse for batched grabs)."""
    if width <= 0 or height <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    region = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
    shot = sct.grab(region)
    arr = np.asarray(shot, dtype=np.uint8)
    return arr[:, :, :3]


def grab_monitor_bgr(monitor_index: int = 1) -> tuple[np.ndarray, dict[str, Any]]:
    """monitor_index 1 = primary in mss. Returns (H,W,3) BGR and monitor dict."""
    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        shot = sct.grab(mon)
        arr = np.asarray(shot, dtype=np.uint8)
        bgr = arr[:, :, :3]
        return bgr, dict(mon)
