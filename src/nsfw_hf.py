"""Adult/NSFW classifier using Hugging Face Transformers (no ONNX export).

This is the "simpler but heavier" option: convenient to prototype, slower to run
and harder to bundle than ONNX.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class AdultHfClassifier:
    """Wraps a HF image-classification pipeline as a score_positive() API."""

    def __init__(self, model_id: str) -> None:
        from transformers import pipeline  # type: ignore[import-not-found]

        # CPU-only by default; user can configure accelerators separately.
        self._pipe = pipeline("image-classification", model=model_id)

    def score_positive(self, bgr_patch: np.ndarray) -> float:
        """Return probability-like score for the most NSFW label found.

        We look for labels containing common NSFW keywords; otherwise we fall back
        to the top label's score.
        """
        if bgr_patch.size == 0:
            return 0.0
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except Exception:
            return 0.0

        rgb = bgr_patch[:, :, ::-1]
        img = Image.fromarray(rgb)
        out: Any = self._pipe(img, top_k=4)
        if not isinstance(out, list) or not out:
            return 0.0

        # Prefer explicit "nsfw" score if present.
        nsfw_score = None
        best_keyword = 0.0
        for item in out:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            if label.strip() == "nsfw":
                nsfw_score = score if nsfw_score is None else max(nsfw_score, score)
            if any(k in label for k in ("nsfw", "adult", "porn", "sexy", "explicit", "nude")):
                best_keyword = max(best_keyword, score)
        if nsfw_score is not None:
            return float(nsfw_score)
        if best_keyword > 0.0:
            return float(best_keyword)

        # Fallback: assume the model's top prediction is meaningful.
        return float(out[0].get("score", 0.0))
