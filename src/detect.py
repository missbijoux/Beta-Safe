"""Heuristic detection of image-like / video-like rectangles on a screen grab.

This is not semantic "photo vs icon" AI; it prefers textured, edge-bounded
regions. Tune thresholds for your desktop. Replace with ONNX/YOLO later.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    score: float


def find_image_like_regions(
    bgr: np.ndarray,
    *,
    max_regions: int = 16,
    min_area_ratio: float = 0.0015,
    min_side: int = 48,
    min_laplacian_var: float = 45.0,
    use_tile_fallback: bool = True,
) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) in same pixel coordinates as *bgr*."""
    if bgr.size == 0:
        return []

    h, w = bgr.shape[:2]
    work_w = min(960, w)
    scale = work_w / float(w)
    work_h = max(1, int(round(h * scale)))
    small = cv2.resize(bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 35, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(small.shape[0] * small.shape[1] * min_area_ratio)

    candidates: list[Region] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue
        if cw < 32 or ch < 32:
            continue
        ar = cw / float(ch)
        if ar < 0.15 or ar > 6.0:
            continue
        roi = small[y : y + ch, x : x + cw]
        if roi.size == 0:
            continue
        lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        if lap_var < min_laplacian_var:
            continue
        # Map back to full-resolution coordinates
        inv = 1.0 / scale
        fx = int(round(x * inv))
        fy = int(round(y * inv))
        fw = int(round(cw * inv))
        fh = int(round(ch * inv))
        fx = max(0, min(fx, w - 1))
        fy = max(0, min(fy, h - 1))
        fw = max(1, min(fw, w - fx))
        fh = max(1, min(fh, h - fy))
        if fw < min_side or fh < min_side:
            continue
        candidates.append(Region(fx, fy, fw, fh, lap_var))

    candidates.sort(key=lambda r: r.score, reverse=True)
    out: list[tuple[int, int, int, int]] = []
    for r in candidates[: max_regions * 2]:
        if _overlaps_too_much(out, (r.x, r.y, r.w, r.h)):
            continue
        out.append((r.x, r.y, r.w, r.h))
        if len(out) >= max_regions:
            break
    if out or not use_tile_fallback:
        return out

    # Fallback: some photos/videos blend into UI and produce weak outer edges.
    # In that case, detect textured/colorful tiles and group them into rectangles.
    return _tile_regions(bgr, max_regions=max_regions, min_side=min_side)


def _tile_regions(
    bgr: np.ndarray,
    *,
    max_regions: int,
    min_side: int,
    tile: int = 64,
    lap_thresh: float = 35.0,
    sat_thresh: float = 35.0,
) -> list[tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    t = max(24, int(tile))
    gh = max(1, h // t)
    gw = max(1, w // t)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    mask = np.zeros((gh, gw), dtype=np.uint8)
    for yi in range(gh):
        y0 = yi * t
        y1 = h if yi == gh - 1 else (yi + 1) * t
        for xi in range(gw):
            x0 = xi * t
            x1 = w if xi == gw - 1 else (xi + 1) * t
            g = gray[y0:y1, x0:x1]
            s = sat[y0:y1, x0:x1]
            if g.size == 0:
                continue
            lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
            sm = float(np.mean(s))
            if lap >= float(lap_thresh) or sm >= float(sat_thresh):
                mask[yi, xi] = 255

    # Group adjacent tiles into connected components, return bounding rectangles.
    mask2 = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask2, connectivity=8)
    rects: list[tuple[int, int, int, int, int]] = []
    for k in range(1, n):
        x, y, ww, hh, area = stats[k]
        if ww <= 0 or hh <= 0:
            continue
        px = x * t
        py = y * t
        pw = min(w - px, ww * t)
        ph = min(h - py, hh * t)
        if pw < min_side or ph < min_side:
            continue
        rects.append((px, py, pw, ph, int(area)))
    rects.sort(key=lambda r: r[4], reverse=True)
    return [(x, y, ww, hh) for (x, y, ww, hh, _a) in rects[:max_regions]]


def tile_candidate_regions(
    bgr: np.ndarray,
    *,
    max_regions: int = 48,
    min_side: int = 48,
    tile: int = 96,
) -> list[tuple[int, int, int, int]]:
    """Public wrapper for the tile-based fallback, useful for adult gating crops."""
    return _tile_regions(bgr, max_regions=max_regions, min_side=min_side, tile=tile)


def grid_tile_regions(
    bgr: np.ndarray,
    *,
    tile: int = 96,
    stride: int | None = None,
    max_regions: int = 96,
    min_side: int = 48,
) -> list[tuple[int, int, int, int]]:
    """Generate a simple overlapping tile grid across the frame.

    This is intentionally dumb but reliable: it guarantees small crops exist even
    when contour-based detection returns one big rectangle.
    """
    if bgr.size == 0:
        return []
    h, w = bgr.shape[:2]
    t = max(min_side, int(tile))
    st = max(min_side, int(stride if stride is not None else (t // 2)))

    rects: list[tuple[int, int, int, int]] = []
    y = 0
    while y < h:
        x = 0
        ph = min(t, h - y)
        if ph < min_side:
            break
        while x < w:
            pw = min(t, w - x)
            if pw < min_side:
                break
            rects.append((x, y, pw, ph))
            if len(rects) >= max_regions:
                return rects
            x += st
        y += st
    return rects


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _overlaps_too_much(existing: list[tuple[int, int, int, int]], cand: tuple[int, int, int, int]) -> bool:
    for e in existing:
        if _iou(e, cand) > 0.55:
            return True
    return False


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy NMS on xyxy boxes (float32). Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        inds = np.where(iou <= iou_thres)[0]
        order = rest[inds]
    return keep


def merge_xywh_nms(
    rects: list[tuple[int, int, int, int]],
    *,
    iou_thresh: float = 0.45,
    max_out: int = 32,
) -> list[tuple[int, int, int, int]]:
    """Merge heuristic + model boxes; drop overlaps via NMS."""
    if not rects:
        return []
    xywh = np.asarray(rects, dtype=np.float32)
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    xyxy = np.stack([x, y, x + w, y + h], axis=1)
    # Prefer keeping smaller boxes when IoUs are high. A near-fullscreen heuristic
    # blob otherwise wins NMS (scores were all equal) and wipes tighter detections.
    areas = np.maximum((xyxy[:, 2] - xyxy[:, 0]).clip(0), 1.0) * np.maximum(
        (xyxy[:, 3] - xyxy[:, 1]).clip(0), 1.0
    )
    scores = (1.0 / areas).astype(np.float32)
    keep = nms_xyxy(xyxy, scores, float(iou_thresh))
    out: list[tuple[int, int, int, int]] = []
    for k in keep:
        x1, y1, x2, y2 = xyxy[k]
        xi, yi = int(x1), int(y1)
        wi = max(1, int(round(x2 - x1)))
        hi = max(1, int(round(y2 - y1)))
        out.append((xi, yi, wi, hi))
        if len(out) >= max_out:
            break
    return out


def filter_xywh_max_area_fraction(
    rects: list[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
    max_area_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """Drop rectangles that cover an implausibly large fraction of the frame."""
    if not rects or frame_w <= 0 or frame_h <= 0:
        return list(rects)
    cap = float(max_area_ratio)
    if cap <= 0.0:
        return []
    if cap >= 1.0:
        return list(rects)
    max_area = float(frame_w) * float(frame_h) * cap
    out: list[tuple[int, int, int, int]] = []
    for x, y, w, h in rects:
        if w <= 0 or h <= 0:
            continue
        if float(w) * float(h) <= max_area:
            out.append((x, y, w, h))
    return out
