"""Which model the ADK agents use. One function, so switching to Gemini is one line.

# OPENROUTER-TEMP: the LiteLlm branch and `log_usage` exist only while there is no
# Gemini key. To remove: delete the `else` branch of `adk_model` (return
# config.SC_MODEL_SMART) and the `after_model_callback=log_usage` arguments in
# app/brain/agents.py.
"""
from __future__ import annotations

import os
from typing import Any

from app import config, gemini


def adk_model() -> Any:
    """Gemini model id when GEMINI_API_KEY is set, else a free OpenRouter model via LiteLLM."""
    if config.provider() == "gemini":
        return config.SC_MODEL_SMART
    # OPENROUTER-TEMP
    from google.adk.models.lite_llm import LiteLlm

    from app import openrouter_backend as orb

    model = orb._require_free(os.getenv("SC_OR_AGENT_MODEL") or orb.models()[0])
    return LiteLlm(model=f"openrouter/{model}", api_base="https://openrouter.ai/api/v1",
                   api_key=orb._key())


def log_usage(callback_context, llm_response):  # OPENROUTER-TEMP
    """after_model_callback: write ADK/LiteLLM usage to jobs/usage.log; refuse non-free models.

    LiteLLM bypasses app/openrouter_backend.py, so this is where the free-only rule
    is enforced for agents: the router reports the model it really used.
    """
    if config.provider() != "openrouter":
        return None
    from app import openrouter_backend as orb

    used = getattr(llm_response, "model_version", None) or "unknown"
    um = getattr(llm_response, "usage_metadata", None)
    if um is not None:
        orb._log_usage(f"adk/{callback_context.agent_name}", used, {
            "prompt_tokens": um.prompt_token_count, "completion_tokens": um.candidates_token_count,
            "cost": None})
    if used != "unknown" and not used.endswith(":free") and used != orb.FREE_ROUTER:
        raise gemini.GeminiUnavailable(f"OpenRouter answered with non-free model {used!r}; stopping")
    return None
