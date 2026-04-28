"""Pixel mosaic and optional blur for BGR image patches."""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def _opencl_ok() -> bool:
    if not config.USE_OPENCL:
        return False
    try:
        if not cv2.ocl.haveOpenCL():
            return False
        cv2.ocl.setUseOpenCL(True)
        return bool(cv2.ocl.useOpenCL())
    except Exception:
        return False


_USE_OCL = _opencl_ok()


def pixel_mosaic(bgr: np.ndarray, block_size: int = 14) -> np.ndarray:
    """Classic mosaic: downscale with area-like shrink, nearest-neighbor upscale."""
    if bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    bs = max(2, int(block_size))
    small_w = max(1, w // bs)
    small_h = max(1, h // bs)

    if _USE_OCL and (w * h) > 40_000:
        try:
            u = cv2.UMat(bgr)
            small = cv2.resize(u, (small_w, small_h), interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            return out.get()
        except Exception:
            pass

    small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def gaussian_blur(bgr: np.ndarray, ksize: int = 31) -> np.ndarray:
    """Heavy blur; ksize must be odd. Same API on Windows and macOS."""
    if bgr.size == 0:
        return bgr
    k = max(3, int(ksize) | 1)

    if _USE_OCL and (bgr.shape[0] * bgr.shape[1]) > 40_000:
        try:
            u = cv2.UMat(bgr)
            out = cv2.GaussianBlur(u, (k, k), 0)
            return out.get()
        except Exception:
            pass

    return cv2.GaussianBlur(bgr, (k, k), 0)
