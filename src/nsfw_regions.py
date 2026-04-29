"""Filter detected rectangles to those likely to be adult / sexualized content."""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np

from . import config
from .nsfw_heuristic import skin_texture_proxy_score


class AdultScoreModel(Protocol):
    def score_positive(self, bgr_patch: np.ndarray) -> float: ...


def adult_filtering_enabled(clf: AdultScoreModel | None) -> bool:
    """True when at least one adult-specific gate is actually available."""
    model_ready = (bool(config.ADULT_ONNX_PATH) or bool(config.ADULT_HF_MODEL)) and clf is not None
    return model_ready or bool(config.ADULT_SKIN_HEURISTIC)


def filter_rects_adult(
    rects: list[tuple[int, int, int, int]],
    bgr_proc: np.ndarray,
    clf: AdultScoreModel | None,
) -> list[tuple[int, int, int, int]]:
    """Return a subset of *rects* to mosaic."""
    if not adult_filtering_enabled(clf):
        return rects
    if (bool(config.ADULT_ONNX_PATH) or bool(config.ADULT_HF_MODEL)) and clf is None and not bool(
        config.ADULT_SKIN_HEURISTIC
    ):
        # Model configured but load failed — do not silently change behavior.
        return rects
    if not rects:
        return rects

    bgr = np.asarray(bgr_proc, dtype=np.uint8)
    h, w = bgr.shape[:2]

    use_model = clf is not None
    use_skin = bool(config.ADULT_SKIN_HEURISTIC)
    combine = (config.ADULT_COMBINE or "any").lower()
    if combine not in ("any", "all"):
        combine = "any"

    ordered = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    max_onnx = int(config.ADULT_MAX_ONNX_CROPS)
    model_indices: set[int] = set()
    if use_model and max_onnx > 0 and ordered:
        if len(ordered) <= max_onnx:
            model_indices = set(range(len(ordered)))
        else:
            # Sample score targets across the full size-ranked list (large + small).
            # Previous logic scored only the first N (largest) rectangles and missed
            # smaller/localized adult regions.
            picks = np.linspace(0, len(ordered) - 1, num=max_onnx, dtype=np.int32)
            model_indices = {int(p) for p in picks.tolist()}

    kept: list[tuple[int, int, int, int]] = []
    debug_adult = os.environ.get("BETASAFE_DEBUG_ADULT", "").strip() not in ("", "0", "false", "off")
    max_model_score = 0.0
    max_skin_score = 0.0
    for j, (x, y, rw, rh) in enumerate(ordered):
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        rw = max(1, min(int(rw), w - x))
        rh = max(1, min(int(rh), h - y))
        patch = bgr[y : y + rh, x : x + rw]
        if patch.size == 0:
            continue

        onnx_s = 0.0
        if use_model and j in model_indices:
            try:
                onnx_s = float(clf.score_positive(patch))
            except Exception:
                onnx_s = 0.0

        skin_s = float(skin_texture_proxy_score(patch)) if use_skin else 0.0
        if onnx_s > max_model_score:
            max_model_score = onnx_s
        if skin_s > max_skin_score:
            max_skin_score = skin_s

        if use_model and use_skin:
            a = onnx_s >= float(config.ADULT_ONNX_THRESHOLD)
            b = skin_s >= float(config.ADULT_SKIN_THRESHOLD)
            ok = (a and b) if combine == "all" else (a or b)
        elif use_model:
            if j in model_indices:
                ok = onnx_s >= float(config.ADULT_ONNX_THRESHOLD)
            else:
                ok = skin_s >= float(config.ADULT_SKIN_THRESHOLD) if use_skin else False
        else:
            ok = skin_s >= float(config.ADULT_SKIN_THRESHOLD)

        if ok:
            kept.append((x, y, rw, rh))

    if debug_adult and (use_model or use_skin):
        print(
            "[betasafe][adult]",
            f"rects_in={len(rects)}",
            f"kept={len(kept)}",
            f"max_model={max_model_score:.3f}",
            f"thr={float(config.ADULT_ONNX_THRESHOLD):.3f}",
            f"max_skin={max_skin_score:.3f}",
            f"skin_thr={float(config.ADULT_SKIN_THRESHOLD):.3f}",
            f"combine={config.ADULT_COMBINE}",
        )

    return kept
