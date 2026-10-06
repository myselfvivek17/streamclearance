"""Per-job timings and model-call counts, written to the job JSON (`timings`, `calls`).

`start(job_id)` puts a Meter in the current context (inherited by asyncio tasks
and asyncio.to_thread workers, like the call budget in app/gemini.py).
`stage(key)` times a block and attributes the Gemini calls made inside it to
`key`; ADK agent calls are attributed by agent name (see app/brain/models.py).

Keys: stage_a, assessor:XX (includes its retrieval), retrieval:XX, verifier:XX,
critic:XX, translation:XX/<language>, total. Stages that run in parallel
overlap, so their times do not add up to `total`.
"""
from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from typing import Any


class Meter:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.timings: dict[str, float] = {}
        self.calls: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def add_time(self, key: str, seconds: float) -> None:
        with self._lock:
            self.timings[key] = round(self.timings.get(key, 0.0) + seconds, 2)

    def add_call(self, key: str, usage: dict[str, int | None], seconds: float | None) -> None:
        with self._lock:
            c = self.calls.setdefault(key, {"calls": 0, "prompt": 0, "out": 0, "thoughts": 0,
                                            "model_s": 0.0})
            c["calls"] += 1
            c["prompt"] += usage.get("prompt") or 0
            c["out"] += usage.get("candidates") or 0
            c["thoughts"] += usage.get("thoughts") or 0
            if seconds is not None:
                c["model_s"] = round(c["model_s"] + seconds, 2)

    def snapshot(self) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        with self._lock:
            return dict(self.timings), {k: dict(v) for k, v in self.calls.items()}


_meter: contextvars.ContextVar[Meter | None] = contextvars.ContextVar("sc_meter", default=None)
_stage: contextvars.ContextVar[str | None] = contextvars.ContextVar("sc_stage", default=None)


def start(job_id: str) -> Meter:
    m = Meter(job_id)
    _meter.set(m)
    return m


def current() -> Meter | None:
    return _meter.get()


def current_stage() -> str | None:
    return _stage.get()


def add_time(key: str, seconds: float) -> None:
    m = _meter.get()
    if m is not None:
        m.add_time(key, seconds)


def record_call(key: str | None, usage: dict[str, int | None], seconds: float | None = None) -> None:
    m = _meter.get()
    if m is not None:
        m.add_call(key or _stage.get() or "other", usage, seconds)


@contextlib.contextmanager
def stage(key: str, *, timed: bool = True):
    """Time this block under `key`; Gemini calls inside it are counted under `key`."""
    token = _stage.set(key)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _stage.reset(token)
        if timed:
            add_time(key, time.perf_counter() - t0)


@contextlib.contextmanager
def timed(key: str):
    """Add this block's time to `key` without changing call attribution."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        add_time(key, time.perf_counter() - t0)
