"""Background worker for adult-gate scoring so the Qt UI thread doesn't freeze.

Important: loading HF models (transformers/torch) can take seconds and must not
run on the Qt UI thread. This worker supports lazy model loading inside the
worker thread on the first job.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Callable

import numpy as np

from .nsfw_regions import AdultScoreModel, filter_rects_adult


@dataclass(frozen=True)
class AdultJob:
    key: int
    rects: list[tuple[int, int, int, int]]
    bgr_proc: np.ndarray


class AdultGateWorker:
    """Single-threaded worker that filters rects using an adult model."""

    def __init__(self, model_loader: Callable[[], AdultScoreModel | None]) -> None:
        self._loader = model_loader
        self._clf: AdultScoreModel | None = None
        self._q: Queue[AdultJob] = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: dict[int, list[tuple[int, int, int, int]]] = {}
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

    def submit(self, *, key: int, rects: list[tuple[int, int, int, int]], bgr_proc: np.ndarray) -> None:
        """Best-effort submit. Drops the job if the worker is busy."""
        try:
            job = AdultJob(key=key, rects=rects, bgr_proc=bgr_proc.copy())
        except Exception:
            return
        try:
            self._q.put_nowait(job)
        except Exception:
            # Busy: skip this frame.
            return

    def get_latest(self, key: int) -> list[tuple[int, int, int, int]] | None:
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
                    self._latest[job.key] = []
                continue
            try:
                t1 = time.time()
                kept = filter_rects_adult(job.rects, job.bgr_proc, self._clf)
            except Exception:
                kept = []
            with self._lock:
                self._latest[job.key] = kept
            if self._debug:
                dt2 = (time.time() - t1) * 1000.0
                print("[betasafe][adult-worker]", f"rects_in={len(job.rects)}", f"kept={len(kept)}", f"ms={dt2:.0f}")
