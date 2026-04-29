"""Fullscreen per-monitor overlay: detect textured rectangles, draw mosaic/blur."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import mss
import numpy as np
import os
from PySide6.QtCore import QObject, QRect, Qt, QTimer, Signal, QRunnable, QThreadPool, Slot
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QGuiApplication, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QStyle, QSystemTrayIcon, QWidget

from . import capture, config, detect, mosaic
from .dxg_capture import DxgiOutputPool
from .adult_worker import AdultGateWorker, frame_signature
from .pipeline_thread import PipelineConfig, PipelineThread
from .nsfw_regions import adult_filtering_enabled, filter_rects_adult


def _repo_root() -> Path:
    """Directory that contains ``src/`` (repo root when running from source)."""
    return Path(__file__).resolve().parent.parent


def _tray_icon() -> QIcon:
    """Prefer packaged branding icons; otherwise a small built-in pixmap (avoids empty tray)."""
    root = _repo_root()
    if sys.platform == "darwin":
        for p in (root / "packaging" / "icons" / "app.icns", root / "packaging" / "icons" / "app.ico"):
            if p.is_file():
                ic = QIcon(str(p))
                if not ic.isNull():
                    return ic
    else:
        for p in (root / "packaging" / "icons" / "app.ico", root / "packaging" / "icons" / "app.icns"):
            if p.is_file():
                ic = QIcon(str(p))
                if not ic.isNull():
                    return ic

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for p in (exe_dir / "app.ico", exe_dir / "app.icns"):
            if p.is_file():
                ic = QIcon(str(p))
                if not ic.isNull():
                    return ic

    app_inst = QApplication.instance()
    if app_inst is not None:
        style = app_inst.style()
        if style is not None:
            pm = style.standardPixmap(QStyle.StandardPixmap.SP_DialogYesButton)
            if not pm.isNull():
                return QIcon(pm)

    pm = QPixmap(64, 64)
    pm.fill(QColor(70, 110, 180))
    painter = QPainter(pm)
    painter.setPen(Qt.GlobalColor.white)
    painter.setFont(QFont("Helvetica", 28, QFont.Weight.Bold))
    painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "B")
    painter.end()
    return QIcon(pm)


def _bgr_patch_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    if h <= 0 or w <= 0:
        return QImage()
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class HotkeyBridge(QObject):
    toggle_pause = Signal()


class CensorOverlay(QWidget):
    """One borderless top-level window aligned to a single QScreen."""

    def __init__(self, screen_index: int) -> None:
        super().__init__()
        self._screen_index = screen_index
        self._running = True
        self._mode = "mosaic"  # "mosaic" | "blur"
        try:
            self._block = max(2, int(os.environ.get("BETASAFE_MOSAIC_BLOCK", "14")))
        except Exception:
            self._block = 14
        self._layers: list[tuple[QRect, QImage]] = []
        self._debug_draw = os.environ.get("BETASAFE_DEBUG_DRAW", "").strip() not in (
            "",
            "0",
            "false",
            "off",
        )
        self._debug_tint = os.environ.get("BETASAFE_DEBUG_TINT", "").strip() not in (
            "",
            "0",
            "false",
            "off",
        )
        try:
            self._hide_delay_ms = max(0, int(os.environ.get("BETASAFE_HIDE_DELAY_MS", "500")))
        except Exception:
            self._hide_delay_ms = 500
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

        # On macOS, Tool windows can end up behind fullscreen/spaces behavior.
        # Prefer a normal top-level Window unless explicitly requested.
        prefer_tool = os.environ.get("BETASAFE_QT_TOOL_WINDOW", "").strip() in (
            "1",
            "true",
            "yes",
            "on",
        )
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        if prefer_tool:
            flags |= Qt.WindowType.Tool
        else:
            flags |= Qt.WindowType.Window
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        except Exception:
            pass

        geo = QApplication.screens()[screen_index].geometry()
        self.setGeometry(geo)

    @property
    def screen_index(self) -> int:
        return self._screen_index

    def censor_active(self) -> bool:
        return self._running

    def effect_mode(self) -> str:
        return self._mode

    def mosaic_block_size(self) -> int:
        return self._block

    def set_paused(self, paused: bool) -> None:
        self._running = not paused
        if paused:
            self._layers.clear()
            self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode if mode in ("mosaic", "blur") else "mosaic"

    def set_click_through(self, on: bool) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, on)
        self.update()

    def set_layers(self, layers: list[tuple[QRect, QImage]]) -> None:
        self._layers = layers
        # Performance/UX: a fullscreen translucent window can cause the compositor
        # to work harder even when we draw nothing. Hide the overlay when there
        # are no layers (unless debug visuals are enabled).
        if not self._layers and not self._debug_draw and not self._debug_tint:
            if self._hide_delay_ms == 0:
                self.hide()
            else:
                if not self._hide_timer.isActive():
                    self._hide_timer.start(self._hide_delay_ms)
            return
        if self._hide_timer.isActive():
            self._hide_timer.stop()
        if not self.isVisible():
            self.show()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if self._debug_tint:
            painter.fillRect(self.rect(), QColor(0, 255, 255, 28))
        for rect, img in self._layers:
            if img.isNull():
                # Fallback: if patch->QImage conversion fails, still draw an opaque
                # block so protection remains visible instead of silently disappearing.
                painter.fillRect(rect, QColor(0, 0, 0, 220))
                if self._debug_draw:
                    painter.setPen(QColor(255, 0, 0, 220))
                    painter.drawRect(rect)
                continue
            painter.drawImage(rect, img)
            if self._debug_draw:
                painter.setPen(QColor(255, 0, 0, 220))
                painter.drawRect(rect)

class _LayerResultSignals(QObject):
    ready = Signal(object, object)  # overlay_key, layers(list[tuple[QRect,QImage]])


class _LayerJob(QRunnable):
    def __init__(
        self,
        *,
        key: int,
        idx: int,
        ww: int,
        wh: int,
        uses_dxgi: bool,
        dxgi_pool: DxgiOutputPool | None,
        rects_proc: list[tuple[int, int, int, int]],
        bgr_proc: np.ndarray,
        mode: str,
        block: int,
        sx: float,
        sy: float,
        signals: _LayerResultSignals,
    ) -> None:
        super().__init__()
        self.key = key
        self.idx = idx
        self.ww = ww
        self.wh = wh
        self.rects_proc = rects_proc
        self.bgr_proc = bgr_proc
        self.mode = mode
        self.block = block
        self.sx = sx
        self.sy = sy
        self.signals = signals

    def run(self) -> None:
        layers: list[tuple[QRect, QImage]] = []
        try:
            for x, y, w, h in self.rects_proc:
                if w <= 2 or h <= 2:
                    continue
                patch = self.bgr_proc[y : y + h, x : x + w]
                if patch.size == 0:
                    continue
                try:
                    if self.mode == "blur":
                        out = mosaic.gaussian_blur(patch, ksize=31)
                    else:
                        out = mosaic.pixel_mosaic(patch, block_size=self.block)
                    img = _bgr_patch_to_qimage(out)
                except Exception:
                    # Skip bad patch but keep the rest of the frame alive.
                    continue
                xd = int(round(x * self.sx))
                yd = int(round(y * self.sy))
                wd = max(1, int(round(w * self.sx)))
                hd = max(1, int(round(h * self.sy)))
                layers.append((QRect(xd, yd, wd, hd), img))
        finally:
            # Always emit so _pending can clear in _apply_layers.
            self.signals.ready.emit(self.key, layers)


class FrameCoordinator(QObject):
    """One timer; DXGI (Windows) + mss fallback; inactive-monitor throttling; ONNX merge."""

    def __init__(self, overlays: list[CensorOverlay], *, enabled: bool = True) -> None:
        super().__init__()
        self._overlays = overlays
        self._use_bg_pipeline = (os.environ.get("BETASAFE_BG_PIPELINE", "").strip() not in ("", "0", "false", "off")) or (
            sys.platform == "darwin"
        )
        self._dxgi: DxgiOutputPool | None = None
        if sys.platform == "win32" and config.USE_DXGI:
            pool = DxgiOutputPool(len(overlays))
            if pool.any_ready():
                self._dxgi = pool

        self._onnx = None
        if config.ONNX_PATH:
            try:
                from .yolov8_onnx import YoloV8OnnxConfig, YoloV8OnnxDetector

                self._onnx = YoloV8OnnxDetector(
                    config.ONNX_PATH,
                    YoloV8OnnxConfig(conf=config.ONNX_CONF, iou=config.ONNX_IOU),
                )
            except Exception:
                self._onnx = None

        # IMPORTANT: do not load HF models on the UI thread (it can freeze on launch).
        self._adult_clf = None

        def _load_adult_model():
            if config.ADULT_HF_MODEL:
                from .nsfw_hf import AdultHfClassifier

                return AdultHfClassifier(config.ADULT_HF_MODEL)
            if config.ADULT_ONNX_PATH:
                from .nsfw_onnx import AdultOnnxClassifier

                return AdultOnnxClassifier(config.ADULT_ONNX_PATH)
            return None

        self._timer = None
        self._frame_counters: dict[int, int] = {}
        self._proc_rect_cache: dict[int, list[tuple[int, int, int, int]]] = {}
        self._adult_rect_cache: dict[int, list[tuple[int, int, int, int]]] = {}
        self._prev_proc: dict[int, np.ndarray | None] = {}
        self._last_layers: dict[int, list[tuple[QRect, QImage]]] = {}
        self._inactive_counters: dict[int, int] = {}
        self._debug = os.environ.get("BETASAFE_DEBUG", "").strip() not in ("", "0", "false", "off")
        self._debug_tick = 0
        self._adult_worker = AdultGateWorker(_load_adult_model)
        self._pool = QThreadPool.globalInstance()
        self._signals = _LayerResultSignals()
        self._signals.ready.connect(self._apply_layers)
        self._pending: set[int] = set()
        self._pipelines: list[PipelineThread] = []

        if not enabled:
            return

        if self._use_bg_pipeline:
            # Run capture+detect+adult+mosaic entirely in background threads.
            for o in self._overlays:
                geo = QApplication.screens()[o.screen_index].geometry()
                pcfg = PipelineConfig(
                    screen_idx=o.screen_index,
                    overlay_key=id(o),
                    cap_left=int(geo.left()),
                    cap_top=int(geo.top()),
                    cap_width=int(geo.width()),
                    cap_height=int(geo.height()),
                    width=max(1, o.width()),
                    height=max(1, o.height()),
                    mode=o.effect_mode(),
                    block=o.mosaic_block_size(),
                )
                p = PipelineThread(cfg=pcfg, load_adult=_load_adult_model)
                p.signals.layers_ready.connect(self._apply_layers)
                p.start()
                self._pipelines.append(p)
        else:
            self._timer = QTimer(self)
            self._timer.setTimerType(Qt.TimerType.PreciseTimer)
            self._timer.setInterval(config.FRAME_INTERVAL_MS)
            self._timer.timeout.connect(self._tick)
            self._timer.start()

    def release_capture(self) -> None:
        if self._dxgi is not None:
            self._dxgi.release()
            self._dxgi = None
        for p in self._pipelines:
            try:
                p.stop()
            except Exception:
                pass

    @property
    def uses_dxgi(self) -> bool:
        return self._dxgi is not None

    @property
    def uses_onnx(self) -> bool:
        return self._onnx is not None

    @property
    def uses_adult_filter(self) -> bool:
        # Enable if configured; readiness is handled in the background worker.
        return bool(config.ADULT_HF_MODEL) or bool(config.ADULT_ONNX_PATH) or bool(config.ADULT_SKIN_HEURISTIC)

    def _tick(self) -> None:
        screens = QApplication.screens()
        at = QGuiApplication.screenAt(QCursor.pos())
        if at is None:
            active_idx = 0
        else:
            try:
                active_idx = screens.index(at)
            except ValueError:
                active_idx = 0

        with mss.mss() as sct:
            for overlay in self._overlays:
                key = id(overlay)
                if not overlay.censor_active():
                    overlay.set_layers([])
                    self._proc_rect_cache.pop(key, None)
                    self._prev_proc.pop(key, None)
                    self._last_layers.pop(key, None)
                    self._adult_rect_cache.pop(key, None)
                    self._frame_counters.pop(key, None)
                    self._inactive_counters.pop(key, None)
                    continue
                idx = overlay.screen_index
                if idx < 0 or idx >= len(screens):
                    overlay.set_layers([])
                    continue

                is_active = idx == active_idx
                if is_active:
                    self._inactive_counters[key] = 0
                else:
                    ie = config.INACTIVE_REFRESH_EVERY
                    if ie > 1:
                        c = self._inactive_counters.get(key, 0) + 1
                        self._inactive_counters[key] = c
                        if (c % ie) != 0:
                            cached = self._last_layers.get(key)
                            if cached is not None:
                                overlay.set_layers(cached)
                                continue

                screen = screens[idx]
                geo = screen.geometry()
                bgr: np.ndarray | None = None
                if self._dxgi is not None:
                    bgr = self._dxgi.grab_bgr(idx)
                if bgr is None or bgr.size == 0:
                    bgr = capture.grab_region_bgr(
                        sct,
                        geo.left(),
                        geo.top(),
                        geo.width(),
                        geo.height(),
                    )
                if bgr.size == 0:
                    overlay.set_layers([])
                    continue

                ww = max(1, overlay.width())
                wh = max(1, overlay.height())
                if bgr.shape[1] != ww or bgr.shape[0] != wh:
                    bgr = cv2.resize(bgr, (ww, wh), interpolation=cv2.INTER_AREA)

                proc_w = min(ww, config.MAX_PROCESS_WIDTH)
                proc_h = max(1, int(round(wh * (proc_w / float(ww)))))
                bgr_proc = cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                proc_area_cap = float(config.MAX_CENSOR_AREA_RATIO)

                prev = self._prev_proc.get(key)
                if prev is not None and prev.shape == bgr_proc.shape:
                    mean_diff = float(
                        np.mean(np.abs(bgr_proc.astype(np.float32) - prev.astype(np.float32)))
                    )
                    if mean_diff < config.IDLE_DIFF_THRESHOLD:
                        cached = self._last_layers.get(key)
                        if cached is not None:
                            overlay.set_layers(cached)
                            continue

                cnt = self._frame_counters.get(key, 0) + 1
                self._frame_counters[key] = cnt
                if cnt % config.DETECT_EVERY_N_FRAMES == 0:
                    rects_h = detect.find_image_like_regions(bgr_proc)
                    rects_o: list[tuple[int, int, int, int]] = []
                    if self._onnx is not None:
                        try:
                            rects_o = self._onnx.infer_xywh(bgr_proc)
                        except Exception:
                            rects_o = []
                    if rects_o:
                        rects_proc = detect.merge_xywh_nms(
                            rects_h + rects_o,
                            iou_thresh=config.MERGE_IOU,
                            max_out=28,
                        )
                    else:
                        rects_proc = rects_h

                    rects_proc = detect.filter_xywh_max_area_fraction(
                        rects_proc, proc_w, proc_h, proc_area_cap
                    )

                    # Adult gating needs smaller/local crops (a butt can be a small part of a big rect).
                    if self.uses_adult_filter and len(rects_proc) < int(config.ADULT_MIN_RECTS):
                        # Prefer a deterministic grid of small crops; it's more likely
                        # to isolate partial nudity than a single huge connected component.
                        # IMPORTANT: keep candidate count bounded; HF/ONNX scoring happens on the UI thread.
                        grid_max = max(12, min(48, int(config.ADULT_MAX_ONNX_CROPS) * 3))
                        rects_grid = detect.grid_tile_regions(
                            bgr_proc,
                            tile=int(config.ADULT_TILE),
                            stride=max(48, int(config.ADULT_TILE) // 2),
                            max_regions=grid_max,
                            min_side=48,
                        )
                        rects_proc = detect.merge_xywh_nms(
                            rects_proc + rects_grid,
                            iou_thresh=0.10,
                            max_out=max(24, grid_max),
                        )
                    rects_proc = detect.filter_xywh_max_area_fraction(
                        rects_proc, proc_w, proc_h, proc_area_cap
                    )
                    self._proc_rect_cache[key] = rects_proc
                else:
                    rects_proc = self._proc_rect_cache.get(key, [])

                before_adult = len(rects_proc)
                if self.uses_adult_filter:
                    # Run expensive HF/ONNX scoring off the UI thread.
                    # Only apply scores that match *this* frame; otherwise stale rects
                    # can mosaic wrong UI (menus/text). If worker is busy or stale,
                    # reuse the last known-good adult-filtered rects to avoid flicker.
                    # Keep a snapshot of candidate rects for the *current* frame.
                    # If adult-worker results are stale (queue lag), we can still
                    # decide whether the kept rects are plausibly relevant by
                    # checking overlap with these candidates.
                    candidates_for_frame = list(rects_proc)

                    sig = frame_signature(bgr_proc)
                    self._adult_worker.submit(
                        key=key, rects=rects_proc, bgr_proc=bgr_proc, sig=sig
                    )
                    pair = self._adult_worker.get_latest(key)
                    if pair is None:
                        rects_proc = self._adult_rect_cache.get(key, [])
                    else:
                        psig, kept = pair
                        if not kept:
                            rects_proc = self._adult_rect_cache.get(key, [])
                        else:
                            # Compute IoU between a kept rect and current candidates.
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

                            overlaps = any(
                                _iou(k, c) > 0.35 for k in kept for c in candidates_for_frame
                            )

                            if psig == sig or overlaps:
                                rects_proc = kept
                                # Update cache when results appear relevant.
                                self._adult_rect_cache[key] = kept
                            else:
                                rects_proc = self._adult_rect_cache.get(key, [])
                after_adult = len(rects_proc)
                rects_proc = detect.filter_xywh_max_area_fraction(
                    rects_proc, proc_w, proc_h, proc_area_cap
                )

                # Build mosaic/blur patches off the UI thread to prevent freezes.
                if key not in self._pending:
                    self._pending.add(key)
                    job = _LayerJob(
                        key=key,
                        idx=idx,
                        ww=ww,
                        wh=wh,
                        uses_dxgi=self._dxgi is not None,
                        dxgi_pool=self._dxgi,
                        rects_proc=rects_proc,
                        bgr_proc=bgr_proc,
                        mode=overlay.effect_mode(),
                        block=overlay.mosaic_block_size(),
                        sx=ww / float(proc_w),
                        sy=wh / float(proc_h),
                        signals=self._signals,
                    )
                    self._pool.start(job)

                self._prev_proc[key] = bgr_proc.copy()

                if self._debug and idx == active_idx:
                    self._debug_tick += 1
                    if self._debug_tick % 20 == 0:
                        print(
                            "[betasafe]",
                            f"screen={idx}",
                            f"rects={before_adult}->{after_adult}",
                            f"layers={len(self._last_layers.get(key, []))}",
                            f"adult_gate={'on' if self.uses_adult_filter else 'off'}",
                            f"hf={'yes' if bool(config.ADULT_HF_MODEL) else 'no'}",
                            f"onnx={'yes' if bool(config.ADULT_ONNX_PATH) else 'no'}",
                            f"adult_worker_ready={'yes' if self._adult_worker.ready else 'no'}",
                        )

    @Slot(object, object)
    def _apply_layers(self, key_obj: object, layers_obj: object) -> None:
        try:
            key = int(key_obj)  # tolerate Signal(object,...)
        except Exception:
            return
        self._pending.discard(key)
        try:
            layers = list(layers_obj)  # type: ignore[arg-type]
        except Exception:
            layers = []
        # Find overlay by id; stable within run.
        for overlay in self._overlays:
            if id(overlay) == key:
                overlay.set_layers(layers)
                self._last_layers[key] = layers
                break


def _start_hotkeys(bridge: HotkeyBridge) -> None:
    try:
        from pynput import keyboard
    except Exception:
        return

    def on_f10() -> None:
        bridge.toggle_pause.emit()

    hk = keyboard.GlobalHotKeys({"<f10>": on_f10})

    def run() -> None:
        hk.run()

    threading.Thread(target=run, name="hotkeys", daemon=True).start()


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(
            None,
            "Tray unavailable",
            "System tray not available; use window controls or run with tray support.",
        )

    # Failsafes to prevent accidental lockups during tuning.
    start_paused = os.environ.get("BETASAFE_START_PAUSED", "").strip() not in ("", "0", "false", "off")

    screens = QApplication.screens()
    overlays: list[CensorOverlay] = []
    if not start_paused:
        for i in range(len(screens)):
            w = CensorOverlay(i)
            w.show()
            overlays.append(w)
    try:
        auto_quit_ms = int(os.environ.get("BETASAFE_AUTO_QUIT_MS", "0").strip() or "0")
    except Exception:
        auto_quit_ms = 0
    if auto_quit_ms > 0:
        QTimer.singleShot(auto_quit_ms, app.quit)

    coordinator = FrameCoordinator(overlays, enabled=not start_paused)
    coordinator.setParent(app)
    app.aboutToQuit.connect(coordinator.release_capture)

    bridge = HotkeyBridge()

    def toggle_pause() -> None:
        for o in overlays:
            o.set_paused(o.censor_active())

    bridge.toggle_pause.connect(toggle_pause)
    _start_hotkeys(bridge)

    tray = QSystemTrayIcon(_tray_icon(), app)
    menu = QMenu()

    act_pause = QAction("Pause / resume (F10)", menu)

    def do_pause() -> None:
        toggle_pause()

    act_pause.triggered.connect(do_pause)

    act_mosaic = QAction("Mode: mosaic", menu)
    act_blur = QAction("Mode: blur", menu)

    def set_all_mode(m: str) -> None:
        for o in overlays:
            o.set_mode(m)

    act_mosaic.triggered.connect(lambda: set_all_mode("mosaic"))
    act_blur.triggered.connect(lambda: set_all_mode("blur"))

    act_click = QAction("Click-through (on)", menu)
    _click_through = True

    def toggle_click() -> None:
        nonlocal _click_through
        _click_through = not _click_through
        for o in overlays:
            o.set_click_through(_click_through)
        act_click.setText("Click-through (on)" if _click_through else "Click-through (off)")

    act_click.triggered.connect(toggle_click)

    act_quit = QAction("Quit", menu)
    act_quit.triggered.connect(app.quit)

    menu.addAction(act_pause)
    menu.addSeparator()
    menu.addAction(act_mosaic)
    menu.addAction(act_blur)
    menu.addSeparator()
    menu.addAction(act_click)
    menu.addSeparator()
    menu.addAction(act_quit)
    tray.setContextMenu(menu)
    caps: list[str] = [f"~{config.TARGET_FPS} FPS", f"{len(overlays)} display(s)"]
    if sys.platform == "win32":
        caps.append("DXGI on" if coordinator.uses_dxgi else "DXGI off (mss)")
    if coordinator.uses_onnx:
        caps.append("ONNX on")
    if coordinator.uses_adult_filter:
        caps.append("adult gate on")
    tray.setToolTip("Betasafe — " + ", ".join(caps) + "; see src/config.py")
    tray.show()

    return app.exec()
