"""The single entry point for every Gemini call (AI Studio key, not Vertex).

Tests mock `generate_json` / `generate_json_with_video`. The API key is read
from the environment and is never printed or logged. Calls are blocking; the
pipeline runs them in worker threads. Rate-limit (429) and transient 5xx
errors are retried with exponential backoff.
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from app import config
from app import openrouter_backend  # OPENROUTER-TEMP

log = logging.getLogger(__name__)

T = TypeVar("T")

RETRY_CODES = {429, 500, 503}
FILE_ACTIVE_TIMEOUT_S = 300


class GeminiUnavailable(RuntimeError):
    pass


_client = None
_sleep = time.sleep  # patched in tests


def _get_client():
    global _client
    if _client is None:
        key = config.gemini_api_key()
        if not key:
            raise GeminiUnavailable("GEMINI_API_KEY is not set (add it to .env)")
        from google import genai

        _client = genai.Client(api_key=key)
    return _client


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    text = str(exc)
    return 429 if "RESOURCE_EXHAUSTED" in text or "429" in text[:40] else None


def with_retry(fn: Callable[[], T], *, what: str, retries: int | None = None) -> T:
    """Run fn, retrying 429/5xx with exponential backoff and jitter."""
    retries = config.GEMINI_MAX_RETRIES if retries is None else retries
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - the SDK raises several types
            code = _status_code(e)
            if code not in RETRY_CODES or attempt == retries:
                raise
            delay = min(60.0, 2.0 * 2**attempt) + random.uniform(0, 1)
            log.warning("gemini %s: HTTP %s, retry %d/%d in %.1fs",
                        what, code, attempt + 1, retries, delay)
            _sleep(delay)
    raise AssertionError("unreachable")


def _config(response_schema: Any, system_instruction: str | None, temperature: float):
    from google.genai import types

    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        system_instruction=system_instruction,
        temperature=temperature,
    )


def _parse(resp: Any, model: str) -> Any:
    if not resp.text:
        raise RuntimeError(f"Empty Gemini response (model={model})")
    return json.loads(resp.text)


def generate_json(
    prompt: str,
    response_schema: Any,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.2,
) -> Any:
    """Call Gemini with structured JSON output and return the parsed JSON."""
    if config.provider() == "openrouter": return openrouter_backend.generate_json(prompt, response_schema, system_instruction=system_instruction, temperature=temperature)  # OPENROUTER-TEMP
    model = model or config.SC_MODEL_SMART
    cfg = _config(response_schema, system_instruction, temperature)
    log.info("gemini.generate_json model=%s prompt_chars=%d", model, len(prompt))
    client = _get_client()
    resp = with_retry(
        lambda: client.models.generate_content(model=model, contents=prompt, config=cfg),
        what="generate_json",
    )
    return _parse(resp, model)


def generate_json_with_video(
    video_path: Path,
    mime_type: str,
    prompt: str,
    response_schema: Any,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.2,
) -> Any:
    """Upload a video via the Files API, wait until ACTIVE, then generate JSON.

    The uploaded file is deleted afterwards (best effort; AI Studio also expires
    files after 48h).
    """
    if config.provider() == "openrouter": return openrouter_backend.generate_json_with_video(video_path, mime_type, prompt, response_schema, system_instruction=system_instruction, temperature=temperature)  # OPENROUTER-TEMP
    model = model or config.SC_MODEL_FAST
    client = _get_client()
    log.info("gemini.files.upload %s (%d bytes)", video_path.name, video_path.stat().st_size)
    f = with_retry(
        lambda: client.files.upload(file=str(video_path), config={"mime_type": mime_type}),
        what="files.upload",
    )
    try:
        deadline = time.monotonic() + FILE_ACTIVE_TIMEOUT_S
        while (state := getattr(f.state, "name", str(f.state))) == "PROCESSING":
            if time.monotonic() > deadline:
                raise RuntimeError("Gemini file processing timed out")
            _sleep(2)
            f = with_retry(lambda: client.files.get(name=f.name), what="files.get")
        if state != "ACTIVE":
            raise RuntimeError(f"Gemini file processing failed (state={state})")

        cfg = _config(response_schema, system_instruction, temperature)
        log.info("gemini.generate_json_with_video model=%s", model)
        resp = with_retry(
            lambda: client.models.generate_content(model=model, contents=[f, prompt], config=cfg),
            what="generate_json_with_video",
        )
        return _parse(resp, model)
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:  # noqa: BLE001
            log.warning("could not delete Gemini file %s", f.name)
