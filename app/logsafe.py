"""Logs carry no content: ids, sizes, counts, durations, model names and token counts only.

Our own log lines (loggers under `app.`) are written that way; uvicorn's lines
hold request paths (ids only) and server events. Library loggers (google-adk,
google-genai, httpx, ...) are not ours to audit and some of them can put a
prompt or a model reply into a message, its arguments or a traceback, so every
library record is reduced to its logger, level and message template (a code
literal, used only when the record came with separate arguments). Tracebacks
are dropped everywhere and replaced by the exception's type name.

Installed as a log-record factory, so it applies to every handler (Cloud Run's
stdout, pytest's caplog, ...).
"""
from __future__ import annotations

import logging

TRUSTED = ("app", "uvicorn")  # audited: their lines carry no user content
WITHHELD = "(message withheld: library text may contain content)"

_installed = False


def _trusted(name: str) -> bool:
    return any(name == t or name.startswith(t + ".") for t in TRUSTED)


def sanitize(record: logging.LogRecord) -> logging.LogRecord:
    exc_type = record.exc_info[0] if record.exc_info and record.exc_info is not True else None
    record.exc_info, record.exc_text, record.stack_info = None, None, None
    if not _trusted(record.name):
        template = record.msg if isinstance(record.msg, str) and record.args else WITHHELD
        record.msg, record.args = template, None
    if exc_type is not None:
        record.msg = f"{record.msg} [{exc_type.__name__}]"
    return record


def install() -> None:
    global _installed
    if _installed:
        return
    base = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        return sanitize(base(*args, **kwargs))

    logging.setLogRecordFactory(factory)
    _installed = True
