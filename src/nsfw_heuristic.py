"""Very weak *proxy* signals (skin-colored pixels + local texture).

This does **not** understand pornography, nudity, or intent. It can flag portraits,
baby photos, medical diagrams, UI with beige tones, sports, dance, etc.

For adult-specific blocking, use a trained classifier ONNX (see ``nsfw_onnx``)
and treat this module as an optional extra gate only if you accept the tradeoffs.
"""

from __future__ import annotations

import cv2
import numpy as np


# Multiple HSV wedges to reduce single-range bias (OpenCV H is 0..179).
_SKIN_RANGES: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((0, 40, 50), (25, 255, 255)),
    ((0, 30, 60), (20, 150, 255)),
    ((160, 40, 50), (179, 255, 255)),
)


def skin_texture_proxy_score(bgr_patch: np.ndarray) -> float:
    """Return ~0..1 where higher means 'more skin-like color coverage *with* texture'."""
    if bgr_patch.size == 0 or bgr_patch.shape[0] < 8 or bgr_patch.shape[1] < 8:
        return 0.0

    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _SKIN_RANGES:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.bitwise_or(mask, m)

    skin_ratio = float(mask.mean() / 255.0)
    gray = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    texture = float(min(1.0, lap_var / 140.0))

    # Down-weight perfectly flat regions (walls, solid fills).
    score = min(1.0, skin_ratio * 2.8) * (0.2 + 0.8 * texture)
    return float(min(1.0, max(0.0, score)))
