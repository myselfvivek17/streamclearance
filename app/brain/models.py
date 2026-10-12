"""Which model the ADK agents use, plus the per-call cost guard and usage log.

ADK builds its own genai client from env vars; `config.apply_adk_env()` points it
at the same backend as app/gemini.py (Vertex AI or AI Studio).
"""
from __future__ import annotations

import time

from app import config, gemini, metrics

# (meter id, agent name) -> start of the agent's model call in flight (one at a time per agent)
_started: dict[tuple[int, str], float] = {}


def stage_of(agent_name: str) -> str:
    """Metrics key of an agent: country_assessor_MY -> assessor:MY, critic_MY -> critic:MY."""
    role, _, cc = agent_name.rpartition("_")
    return f"{role.removeprefix('country_')}:{cc}" if role else agent_name


def adk_model() -> str:
    """Gemini model id for the ADK agents."""
    config.apply_adk_env()
    return config.SC_MODEL_SMART


def agent_config(thinking: str | None = None):
    """Generation settings for the ADK agents: low temperature and a fixed seed, so the
    same video gets the same assessment (config.JUDGE_TEMPERATURE, SC_MODEL_SEED).
    thinking: Gemini thinking level, None = the model's default."""
    from google.genai import types

    return types.GenerateContentConfig(
        temperature=config.JUDGE_TEMPERATURE, seed=config.MODEL_SEED,
        thinking_config=types.ThinkingConfig(thinking_level=thinking.upper()) if thinking else None)


def count_call(callback_context, llm_request):
    """before_model_callback: refuse the call once the job's call budget is used up."""
    gemini.count_call(f"adk/{callback_context.agent_name}")
    _started[(id(metrics.current()), callback_context.agent_name)] = time.perf_counter()
    return None


def log_usage(callback_context, llm_response):
    """after_model_callback: write the call's token counts to jobs/usage.log."""
    name = callback_context.agent_name
    t0 = _started.pop((id(metrics.current()), name), None)
    gemini.log_usage(f"adk/{name}",
                     getattr(llm_response, "model_version", None) or config.SC_MODEL_SMART,
                     getattr(llm_response, "usage_metadata", None), stage=stage_of(name),
                     seconds=None if t0 is None else time.perf_counter() - t0)
    return None
