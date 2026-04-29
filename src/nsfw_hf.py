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
        out: Any = self._pipe(img, top_k=8)
        if not isinstance(out, list) or not out:
            return 0.0

        label_scores: dict[str, float] = {}
        for item in out:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).lower().strip()
            if not label:
                continue
            score = float(item.get("score", 0.0))
            label_scores[label] = max(label_scores.get(label, 0.0), score)

        if "nsfw" in label_scores:
            return float(label_scores["nsfw"])

        nsfw_like = (
            "nsfw",
            "adult",
            "porn",
            "sexy",
            "explicit",
            "nude",
            "hentai",
            "erotic",
        )
        best_kw = 0.0
        for lab, sc in label_scores.items():
            if any(k in lab for k in nsfw_like):
                best_kw = max(best_kw, sc)
        if best_kw > 0.0:
            return float(best_kw)

        # Binary export (e.g. normal vs nsfw): return the *unsafe* class score, never blindly
        # use top-1 (often "normal" with the highest probability on safe UI).
        if len(label_scores) == 2:
            labs = list(label_scores.keys())
            a, b = labs[0], labs[1]
            safe_tokens = ("normal", "safe", "neutral", "sfw", "drawing", "clean", "benign")
            a_safe = any(t in a for t in safe_tokens)
            b_safe = any(t in b for t in safe_tokens)
            if a_safe and not b_safe:
                return float(label_scores[b])
            if b_safe and not a_safe:
                return float(label_scores[a])

        return 0.0
