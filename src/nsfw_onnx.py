"""Binary (or small multi-class) NSFW / adult-content ONNX classifiers via onnxruntime.

Export your own model or convert an open weights checkpoint to ONNX. Input is
assumed NCHW float32 RGB in ``[0, 1]`` after resizing to the network's fixed
height/width (common for MobileNet-style heads).

Output parsing (first output tensor only):

- shape ``(1,)`` or ``(1,1)``: treated as logit if outside ``[0,1]``, else probability
- shape ``(1,2)`` or ``(2,)``: softmax; ``BETASAFE_ADULT_ONNX_POS_CLASS`` picks which
  index is the 'positive' (typically 1 = NSFW).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import config


class AdultOnnxClassifier:
    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as ort  # type: ignore[import-not-found]

        self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._inp = self._session.get_inputs()[0]
        shp = self._inp.shape
        self._in_h = self._static_dim(shp[2], 224)
        self._in_w = self._static_dim(shp[3], self._in_h)
        self._pos_index = int(config.ADULT_ONNX_POS_CLASS)

    @staticmethod
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

    def score_positive(self, bgr_patch: np.ndarray) -> float:
        """Higher => more confident the patch matches the model's 'positive' class."""
        if bgr_patch.size == 0:
            return 0.0
        img = cv2.resize(bgr_patch, (self._in_w, self._in_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        out = self._session.run(None, {self._inp.name: blob})[0]
        return _parse_positive_probability(np.asarray(out, dtype=np.float64), self._pos_index)


def _parse_positive_probability(raw: np.ndarray, pos_index: int) -> float:
    x = np.reshape(raw, (-1,))
    if x.size == 1:
        v = float(x[0])
        if 0.0 <= v <= 1.0:
            return v
        return float(1.0 / (1.0 + math.exp(-v)))
    if x.size == 2:
        e = np.exp(x - np.max(x))
        p = e / (np.sum(e) + 1e-9)
        idx = 1 if pos_index not in (0, 1) else pos_index
        return float(p[idx])
    # Fallback: treat as logits over small vocab — use max softmax chunk if looks batched
    if x.size > 2:
        e = np.exp(x - np.max(x))
        p = e / (np.sum(e) + 1e-9)
        idx = max(0, min(int(pos_index), p.size - 1))
        return float(p[idx])
    return 0.0
