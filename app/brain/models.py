"""Which model the ADK agents use, plus the per-call cost guard and usage log.

ADK builds its own genai client from env vars; `config.apply_adk_env()` points it
at the same backend as app/gemini.py (Vertex AI or AI Studio).
"""
from __future__ import annotations

from app import config, gemini


def adk_model() -> str:
    """Gemini model id for the ADK agents."""
    config.apply_adk_env()
    return config.SC_MODEL_SMART


def count_call(callback_context, llm_request):
    """before_model_callback: refuse the call once the job's call budget is used up."""
    gemini.count_call(f"adk/{callback_context.agent_name}")
    return None


def log_usage(callback_context, llm_response):
    """after_model_callback: write the call's token counts to jobs/usage.log."""
    gemini.log_usage(f"adk/{callback_context.agent_name}",
                     getattr(llm_response, "model_version", None) or config.SC_MODEL_SMART,
                     getattr(llm_response, "usage_metadata", None))
    return None
