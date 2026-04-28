"""Windows DXGI screen capture via dxcam (per physical output). Falls back to mss when unavailable."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np


def _import_dxcam() -> Any | None:
    if sys.platform != "win32":
        return None
    try:
        import dxcam  # type: ignore[import-not-found]

        return dxcam
    except Exception:
        return None


class DxgiOutputPool:
    """One DXCamera per Qt screen index (best-effort: device 0, output i)."""

    def __init__(self, num_screens: int) -> None:
        self._cams: list[Any | None] = [None] * max(0, num_screens)
        self._dxcam = _import_dxcam()
        if self._dxcam is None:
            return
        for i in range(num_screens):
            try:
                self._cams[i] = self._dxcam.create(
                    device_idx=0,
                    output_idx=i,
                    output_color="BGR",
                    backend="dxgi",
                    processor_backend="cv2",
                )
            except Exception:
                self._cams[i] = None

    def any_ready(self) -> bool:
        return any(c is not None for c in self._cams)

    def grab_bgr(self, screen_idx: int) -> np.ndarray | None:
        if screen_idx < 0 or screen_idx >= len(self._cams):
            return None
        cam = self._cams[screen_idx]
        if cam is None:
            return None
        try:
            frame = cam.grab()
        except Exception:
            return None
        if frame is None:
            return None
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        return np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)

    def release(self) -> None:
        for cam in self._cams:
            if cam is None:
                continue
            try:
                cam.release()
            except Exception:
                pass
        self._cams = []
