"""Optional YOLOv8 ONNX detector (onnxruntime). Export with Ultralytics: yolo export model=yolov8n.pt format=onnx."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import detect


def _letterbox(bgr: np.ndarray, new_size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = bgr.shape[:2]
    r = min(new_size / h, new_size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw = (new_size - nw) / 2.0
    dh = (new_size - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (float(dw), float(dh))


def _scale_boxes_to_orig(
    boxes_xyxy: np.ndarray,
    *,
    img1_hw: tuple[int, int],
    img0_hw: tuple[int, int],
) -> np.ndarray:
    """Map boxes from letterboxed network input (img1) back to original (img0) pixel coords."""
    gain = min(img1_hw[0] / img0_hw[0], img1_hw[1] / img0_hw[1])
    pad_w = (img1_hw[1] - img0_hw[1] * gain) / 2.0
    pad_h = (img1_hw[0] - img0_hw[0] * gain) / 2.0
    b = boxes_xyxy.copy().astype(np.float32)
    b[:, [0, 2]] -= pad_w
    b[:, [1, 3]] -= pad_h
    b /= gain
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, img0_hw[1] - 1)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, img0_hw[0] - 1)
    return b


@dataclass
class YoloV8OnnxConfig:
    conf: float = 0.35
    iou: float = 0.45


def _static_dim(val: object, fallback: int) -> int:
    if isinstance(val, int) and val > 0:
        return val
    try:
        n = int(val)
        if n > 0:
            return n
    except Exception:
        pass
    return fallback


class YoloV8OnnxDetector:
    def __init__(self, onnx_path: str, cfg: YoloV8OnnxConfig) -> None:
        import onnxruntime as ort  # type: ignore[import-not-found]

        self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._cfg = cfg
        self._inp = self._session.get_inputs()[0]
        shp = self._inp.shape
        self._in_h = _static_dim(shp[2], 640)
        self._in_w = _static_dim(shp[3], self._in_h)

    def infer_xywh(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        if bgr.size == 0:
            return []
        h0, w0 = bgr.shape[:2]
        in_size = max(self._in_h, self._in_w)
        lb, _r, _pad = _letterbox(bgr, in_size)
        img = lb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]  # 1,3,H,W
        out = self._session.run(None, {self._inp.name: img})[0]
        if out.ndim != 3 or min(out.shape[1], out.shape[2]) < 4:
            return []
        # (1, 84, N) or (1, N, 84)
        if out.shape[1] == 84:
            pred = out[0].T
        elif out.shape[2] == 84:
            pred = out[0]
        else:
            return []
        xywh = pred[:, :4].astype(np.float32)
        cls_scores = pred[:, 4:]
        scores = cls_scores.max(axis=1)
        mask = scores >= float(self._cfg.conf)
        if not np.any(mask):
            return []
        xywh = xywh[mask]
        scores = scores[mask]
        cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        ih1, iw1 = lb.shape[:2]
        boxes = _scale_boxes_to_orig(boxes, img1_hw=(ih1, iw1), img0_hw=(h0, w0))
        keep = detect.nms_xyxy(boxes, scores.astype(np.float32), float(self._cfg.iou))
        rects: list[tuple[int, int, int, int]] = []
        for k in keep:
            x1, y1, x2, y2 = boxes[k]
            x, y, w, h = int(x1), int(y1), max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))
            rects.append((x, y, w, h))
        return rects
