"""The job's activity feed: what the pipeline actually did, step by step (job `events`).

Every event comes from a real pipeline point (stage A finished, a law search the
assessor made, the critic's verdicts, a fallback, a translation result); nothing
is scripted. `start(job_id)` puts an EventLog in the current context, inherited
by asyncio tasks and asyncio.to_thread workers (like app/metrics.py), so code
deep in the pipeline calls `emit(...)` without knowing about jobs.

Event text quotes the video (transcript, on-screen text) and model output, so it
is report data: stored in the job record, deleted with it, and never logged
(app/logsafe.py rules; tests/test_logs.py checks a canary never reaches a log).

Event: {"t": seconds since the job started, "stage": "inventory"|"ai"|"country"|
"translation"|"report", "country": "MY" (optional), "kind": see KINDS, "text": str,
"ref": provision id or time range (optional)}.
"""
from __future__ import annotations

import contextvars
import threading
import time
from typing import Any

MAX_EVENTS = 200
# watch: what stage A saw/heard; search: a law search or lookup; law: provisions found or
# assessed; review: the critic or verifier; translate: the translation check;
# ai: the AI-generated content check; warn: a fallback, a failure, something to check by hand;
# info: summaries.
KINDS = ("watch", "search", "law", "review", "translate", "ai", "warn", "info")
TEXT_MAX = 300


class EventLog:
    def __init__(self, job_id: str, t0: float | None = None):
        self.job_id = job_id
        self.t0 = time.time() if t0 is None else t0
        self.events: list[dict[str, Any]] = []
        self.dropped = 0
        self.version = 0  # bumps on every change, so the job is saved only when needed
        self._lock = threading.Lock()

    def add(self, kind: str, text: str, *, stage: str, country: str | None = None,
            ref: str | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        text = " ".join(str(text).split())
        if len(text) > TEXT_MAX:
            text = text[: TEXT_MAX - 1] + "…"
        ev: dict[str, Any] = {"t": round(max(0.0, time.time() - self.t0), 1), "stage": stage,
                              "kind": kind, "text": text}
        if country:
            ev["country"] = country
        if ref:
            ev["ref"] = ref
        with self._lock:
            if len(self.events) >= MAX_EVENTS - 1:
                self.dropped += 1
                note = f"{self.dropped} more step(s) not shown (the feed keeps {MAX_EVENTS})."
                if len(self.events) == MAX_EVENTS - 1:
                    self.events.append({"t": ev["t"], "stage": stage, "kind": "info", "text": note})
                else:
                    self.events[-1] = {**self.events[-1], "t": ev["t"], "text": note}
            else:
                self.events.append(ev)
            self.version += 1

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self.events]


_log: contextvars.ContextVar[EventLog | None] = contextvars.ContextVar("sc_events", default=None)


def start(job_id: str, t0: float | None = None) -> EventLog:
    log = EventLog(job_id, t0)
    _log.set(log)
    return log


def current() -> EventLog | None:
    return _log.get()


def emit(kind: str, text: str, *, stage: str, country: str | None = None,
         ref: str | None = None) -> None:
    """Add an event to the current job's feed (no-op outside a job, e.g. in unit tests)."""
    log = _log.get()
    if log is not None:
        log.add(kind, text, stage=stage, country=country, ref=ref)


def clock(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"


def span(start: float, end: float) -> str:
    return f"{clock(start)}–{clock(end)}"


def quote(text: str, n: int = 80) -> str:
    text = " ".join((text or "").split())
    return f"“{text[:n]}{'…' if len(text) > n else ''}”"
