"""Background pipeline per monitor: capture -> downscale -> detect -> adult gate -> mosaic.

Purpose: keep the Qt UI thread responsive on macOS by doing all heavy work in a
Python background thread, emitting only ready-to-draw layers.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import cv2
import mss
import numpy as np
from PySide6.QtCore import QObject, QRect, Signal
from PySide6.QtGui import QImage

from . import capture, config, detect, mosaic
from .nsfw_regions import AdultScoreModel, filter_rects_adult


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    if h <= 0 or w <= 0:
        return QImage()
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class PipelineSignals(QObject):
    layers_ready = Signal(object, object)  # overlay_key, layers


@dataclass(frozen=True)
class PipelineConfig:
    screen_idx: int
    overlay_key: int
    cap_left: int
    cap_top: int
    cap_width: int
    cap_height: int
    width: int
    height: int
    mode: str
    block: int


class PipelineThread:
    def __init__(self, *, cfg: PipelineConfig, load_adult: Callable[[], AdultScoreModel | None]) -> None:
        self._cfg = cfg
        self._load_adult = load_adult
        self.signals = PipelineSignals()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name=f"pipeline-{cfg.screen_idx}", daemon=True)
        self._adult: AdultScoreModel | None = None
        self._debug = os.environ.get("BETASAFE_DEBUG_PIPELINE", "").strip() not in ("", "0", "false", "off")

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            if self._debug:
                print(
                    "[betasafe][pipeline] start",
                    f"screen={self._cfg.screen_idx}",
                    f"cap=({self._cfg.cap_left},{self._cfg.cap_top},{self._cfg.cap_width},{self._cfg.cap_height})",
                    f"out=({self._cfg.width}x{self._cfg.height})",
                    flush=True,
                )
            if config.ADULT_HF_MODEL or config.ADULT_ONNX_PATH:
                try:
                    self._adult = self._load_adult()
                except Exception:
                    self._adult = None

            interval = max(0.01, config.FRAME_INTERVAL_MS / 1000.0)
            prev_proc: np.ndarray | None = None
            rects_cache: list[tuple[int, int, int, int]] = []
            frame = 0

            with mss.mss() as sct:
                while not self._stop.is_set():
                    t0 = time.time()
                    frame += 1

                    # Capture this monitor region (virtual-desktop coords).
                    if self._debug and frame % 30 == 1:
                        print("[betasafe][pipeline] grabbing…", f"screen={self._cfg.screen_idx}", flush=True)
                    bgr = capture.grab_region_bgr(
                        sct,
                        self._cfg.cap_left,
                        self._cfg.cap_top,
                        self._cfg.cap_width,
                        self._cfg.cap_height,
                    )
                    if self._debug and frame % 30 == 1:
                        print("[betasafe][pipeline] grabbed", f"shape={getattr(bgr, 'shape', None)}", flush=True)
                    if bgr.size == 0:
                        time.sleep(interval)
                        continue

                    # Resize to overlay logical size.
                    if bgr.shape[1] != self._cfg.width or bgr.shape[0] != self._cfg.height:
                        bgr = cv2.resize(
                            bgr, (self._cfg.width, self._cfg.height), interpolation=cv2.INTER_AREA
                        )

                    ww, wh = self._cfg.width, self._cfg.height
                    proc_w = min(ww, config.MAX_PROCESS_WIDTH)
                    proc_h = max(1, int(round(wh * (proc_w / float(ww)))))
                    bgr_proc = cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                    proc_area_cap = float(config.MAX_CENSOR_AREA_RATIO)

                    # Idle diff shortcut.
                    if prev_proc is not None and prev_proc.shape == bgr_proc.shape:
                        mean_diff = float(
                            np.mean(np.abs(bgr_proc.astype(np.float32) - prev_proc.astype(np.float32)))
                        )
                        if mean_diff < config.IDLE_DIFF_THRESHOLD:
                            time.sleep(max(0.0, interval - (time.time() - t0)))
                            continue

                    # Detect every N frames.
                    if frame % config.DETECT_EVERY_N_FRAMES == 0:
                        rects = detect.find_image_like_regions(bgr_proc)
                        # Add grid tiles when adult gate configured.
                        if (
                            config.ADULT_HF_MODEL
                            or config.ADULT_ONNX_PATH
                            or config.ADULT_SKIN_HEURISTIC
                        ) and len(rects) < int(config.ADULT_MIN_RECTS):
                            grid_max = max(12, min(48, int(config.ADULT_MAX_ONNX_CROPS) * 3))
                            rects += detect.grid_tile_regions(
                                bgr_proc,
                                tile=int(config.ADULT_TILE),
                                stride=max(48, int(config.ADULT_TILE) // 2),
                                max_regions=grid_max,
                                min_side=48,
                            )
                            rects = detect.merge_xywh_nms(
                                rects, iou_thresh=0.10, max_out=max(24, grid_max)
                            )
                        rects_cache = rects
                    else:
                        rects = rects_cache

                    rects = detect.filter_xywh_max_area_fraction(
                        rects, proc_w, proc_h, proc_area_cap
                    )

                    # Adult gate (runs in this background thread).
                    rects = filter_rects_adult(rects, bgr_proc, self._adult)
                    rects = detect.filter_xywh_max_area_fraction(
                        rects, proc_w, proc_h, proc_area_cap
                    )

                    # Build layers.
                    sx = ww / float(proc_w)
                    sy = wh / float(proc_h)
                    layers: list[tuple[QRect, QImage]] = []
                    for x, y, w, h in rects[: int(config.ADULT_MAX_ONNX_CROPS) * 4]:
                        patch = bgr_proc[y : y + h, x : x + w]
                        if self._cfg.mode == "blur":
                            out = mosaic.gaussian_blur(patch, ksize=31)
                        else:
                            out = mosaic.pixel_mosaic(patch, block_size=self._cfg.block)
                        img = _bgr_to_qimage(out)
                        xd = int(round(x * sx))
                        yd = int(round(y * sy))
                        wd = max(1, int(round(w * sx)))
                        hd = max(1, int(round(h * sy)))
                        layers.append((QRect(xd, yd, wd, hd), img))

                    self.signals.layers_ready.emit(self._cfg.overlay_key, layers)
                    prev_proc = bgr_proc

                    if self._debug and frame % 30 == 0:
                        print(
                            "[betasafe][pipeline]",
                            f"screen={self._cfg.screen_idx}",
                            f"rects={len(rects)}",
                            flush=True,
                        )

                    dt = time.time() - t0
                    time.sleep(max(0.0, interval - dt))
        except Exception:
            # Never let the pipeline thread die silently.
            if self._debug or os.environ.get("BETASAFE_DEBUG_PIPELINE", "").strip() not in ("", "0", "false", "off"):
                print("[betasafe][pipeline] crashed", flush=True)
                traceback.print_exc()
