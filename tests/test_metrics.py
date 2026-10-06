import asyncio

from app import gemini, metrics
from app.brain.models import stage_of


class _Usage:
    prompt_token_count, candidates_token_count, thoughts_token_count, total_token_count = 10, 2, 5, 17


def test_calls_attributed_to_current_stage_and_explicit_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(gemini, "USAGE_LOG", tmp_path / "usage.log")

    async def job():
        m = metrics.start("j1")

        async def country(cc):
            with metrics.stage(f"translation:{cc}/X"):
                await asyncio.to_thread(gemini.log_usage, "generate_json", "m", _Usage(), seconds=0.5)

        await asyncio.gather(country("MY"), country("IN"))
        gemini.log_usage("adk/critic_MY", "m", _Usage(), stage=stage_of("critic_MY"))
        return m.snapshot()

    timings, calls = asyncio.run(job())
    assert set(timings) == {"translation:MY/X", "translation:IN/X"}
    assert calls["translation:MY/X"] == {"calls": 1, "prompt": 10, "out": 2, "thoughts": 5,
                                         "model_s": 0.5}
    assert calls["critic:MY"]["calls"] == 1


def test_stage_of_agent_names():
    assert stage_of("country_assessor_SG") == "assessor:SG"
    assert stage_of("critic_IN") == "critic:IN"
