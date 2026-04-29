"""Background worker for adult-gate scoring so the Qt UI thread doesn't freeze.

Important: loading HF models (transformers/torch) can take seconds and must not
run on the Qt UI thread. This worker supports lazy model loading inside the
worker thread on the first job.
"""

from __future__ import annotations

import os
import threading
import time
import zlib
from dataclasses import dataclass
from queue import Full, Queue
from typing import Callable

import numpy as np

from .nsfw_regions import AdultScoreModel, filter_rects_adult


def frame_signature(bgr: np.ndarray) -> int:
    """Cheap fingerprint of *bgr* so UI thread can ignore stale adult-gate results."""
    if bgr.size == 0:
        return 0
    step_y = max(1, bgr.shape[0] // 48)
    step_x = max(1, bgr.shape[1] // 48)
    sample = bgr[::step_y, ::step_x, :]
    return int(zlib.crc32(sample.tobytes()) & 0xFFFFFFFF)


@dataclass(frozen=True)
class AdultJob:
    key: int
    rects: list[tuple[int, int, int, int]]
    bgr_proc: np.ndarray
    sig: int


class AdultGateWorker:
    """Single-threaded worker that filters rects using an adult model."""

    def __init__(self, model_loader: Callable[[], AdultScoreModel | None]) -> None:
        self._loader = model_loader
        self._clf: AdultScoreModel | None = None
        # Small buffer so a slow HF frame doesn't always drop the next submission.
        self._q: Queue[AdultJob] = Queue(maxsize=2)
        self._lock = threading.Lock()
        # overlay id -> (signature of bgr_proc used for scoring, rects to mosaic)
        self._latest: dict[int, tuple[int, list[tuple[int, int, int, int]]]] = {}
        self._ready = False
        self._debug = os.environ.get("BETASAFE_DEBUG_ADULT_WORKER", "").strip() not in (
            "",
            "0",
            "false",
            "off",
        )
        self._t = threading.Thread(target=self._run, name="adult-gate-worker", daemon=True)
        self._t.start()

    @property
    def ready(self) -> bool:
        return self._ready and self._clf is not None

    def submit(
        self,
        *,
        key: int,
        rects: list[tuple[int, int, int, int]],
        bgr_proc: np.ndarray,
        sig: int,
    ) -> bool:
        """Queue adult scoring for this frame. Returns False if worker is busy (job dropped)."""
        try:
            job = AdultJob(key=key, rects=rects, bgr_proc=bgr_proc.copy(), sig=sig)
        except Exception:
            return False
        try:
            self._q.put_nowait(job)
        except Full:
            return False
        return True

    def get_latest(self, key: int) -> tuple[int, list[tuple[int, int, int, int]]] | None:
        with self._lock:
            return self._latest.get(key)

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if self._clf is None:
                if self._debug:
                    print("[betasafe][adult-worker] loading model…")
                t0 = time.time()
                try:
                    self._clf = self._loader()
                except Exception:
                    self._clf = None
                self._ready = True
                if self._debug:
                    dt = (time.time() - t0) * 1000.0
                    print(
                        "[betasafe][adult-worker]",
                        "model_loaded" if self._clf is not None else "model_failed",
                        f"ms={dt:.0f}",
                    )
            if self._clf is None:
                with self._lock:
                    self._latest[job.key] = (job.sig, [])
                continue
            try:
                t1 = time.time()
                kept = filter_rects_adult(job.rects, job.bgr_proc, self._clf)
            except Exception:
                kept = []
            with self._lock:
                self._latest[job.key] = (job.sig, kept)
            if self._debug:
                dt2 = (time.time() - t1) * 1000.0
                print("[betasafe][adult-worker]", f"rects_in={len(job.rects)}", f"kept={len(kept)}", f"ms={dt2:.0f}")
