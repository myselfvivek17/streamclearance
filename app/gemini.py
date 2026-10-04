"""The single entry point for every Gemini call (AI Studio key, not Vertex).

Tests mock `generate_json`. The API key is read from the environment and is
never printed or logged.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app import config

log = logging.getLogger(__name__)


class GeminiUnavailable(RuntimeError):
    pass


_client = None


def _get_client():
    global _client
    if _client is None:
        key = config.gemini_api_key()
        if not key:
            raise GeminiUnavailable("GEMINI_API_KEY is not set (add it to .env)")
        from google import genai

        _client = genai.Client(api_key=key)
    return _client


def generate_json(
    prompt: str,
    response_schema: Any,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.2,
) -> Any:
    """Call Gemini with structured JSON output and return the parsed JSON."""
    from google.genai import types

    model = model or config.SC_MODEL_SMART
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        system_instruction=system_instruction,
        temperature=temperature,
    )
    log.info("gemini.generate_json model=%s prompt_chars=%d", model, len(prompt))
    resp = _get_client().models.generate_content(model=model, contents=prompt, config=cfg)
    if not resp.text:
        raise RuntimeError(f"Empty Gemini response (model={model})")
    return json.loads(resp.text)
