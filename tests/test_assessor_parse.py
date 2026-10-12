"""Assessor reliability: tolerant parsing of the agent's reply, and no silent fallback."""
import asyncio
import json

import pytest

from app import events, gemini, pipeline
from app.brain import agents, assess_country
from app.brain.agents import CriticVerdict, MomentAssessment, _parse_list

A = {"start_s": 3.5, "end_s": 11.5, "leaf": "health.cure_claim", "covered_by": ["X"], "risk": "red"}
B = {"start_s": 11.5, "end_s": 18.0, "leaf": "gambling.promotion", "covered_by": ["Y"], "risk": "amber"}
INNER = lambda d: {k: v for k, v in d.items() if k not in ("start_s", "end_s")}  # noqa: E731


@pytest.mark.parametrize("shape", [
    [A, B],
    [[A], [B]],  # one list per moment: was "dropping malformed MomentAssessment: model_type" x N
    {"assessments": [A, B]},
    {"assessments": [A, B], "summary": ["two moments"]},
    [json.dumps(A), json.dumps(B)],  # objects encoded as strings
    {"0": A, "1": B},
    [{"start_s": 3.5, "end_s": 11.5, "assessments": [INNER(A)]},
     {"moment": {"start_s": 11.5, "end_s": 18.0}, "assessments": [INNER(B)]}],
    {"result": {"assessments": [A, B]}},
])
def test_known_shapes_parse(shape):
    out = _parse_list("Here is my answer:\n```json\n" + json.dumps(shape) + "\n```", MomentAssessment)
    assert [(a.start_s, a.leaf, a.covered_by) for a in out] == [
        (3.5, "health.cure_claim", ["X"]), (11.5, "gambling.promotion", ["Y"])]


def test_lenient_fields_and_unknown_keys():
    out = _parse_list(json.dumps([{**A, "risk": "Red", "covered_by": "X", "confidence": 0.9}]), MomentAssessment)
    assert out[0].risk == "red" and out[0].covered_by == ["X"]
    assert _parse_list('[{"index": 0, "action": "Downgrade", "risk": "AMBER"}]', CriticVerdict)[0].risk == "amber"


def test_malformed_items_logged_by_field_name_only(caplog):
    out = _parse_list(json.dumps([A, {"leaf": "secret words here"}, 42]), MomentAssessment)
    assert len(out) == 1
    msgs = [r.getMessage() for r in caplog.records if "dropping malformed" in r.getMessage()]
    assert len(msgs) == 2 and all("secret" not in m for m in msgs)
    assert any("start_s" in m for m in msgs)


def test_no_json_raises():
    with pytest.raises(ValueError):
        _parse_list("I could not decide.", MomentAssessment)


def _inventory():
    return pipeline.mock_inventory(26.0)


@pytest.mark.parametrize("reply, reason", [("I could not decide.", "no_json"),
                                           ('[{"note": "nothing"}]', "no_valid_items")])
def test_fallback_is_recorded_not_silent(real_provisions, monkeypatch, reply, reason):
    async def fake_agent(agent, message):
        return reply

    inv = _inventory()
    seg = next(s for s in inv.segments if "health.cure_claim" in s.leaves)
    monkeypatch.setattr(agents, "run_agent", fake_agent)
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **kw: [
        {"start_s": seg.start_s, "end_s": seg.end_s, "leaf": "health.cure_claim",
         "covered_by": ["MY/MAS1956/s3(1)"], "risk": "red", "rationale": "r", "suggested_fix": "f"}])
    feed = events.start("job")

    async def run():
        res = await agents.assess(inv, "MY", mock=False)
        assert res.method == "fallback" and res.fallback_reason == reason
        assert res.notes[0].startswith("Simplified check:")
        return await assess_country(inv, "MY", mock=True)  # report plumbing below

    asyncio.run(run())
    assert any(e["kind"] == "warn" and "simplified check" in e["text"] for e in feed.snapshot())


def test_report_carries_method(real_provisions, monkeypatch):
    async def fake_assess(inventory, country, *, mock):
        return agents.Assessed(agents.mock_assess(inventory, country), ["Simplified check: x"],
                               method="fallback", fallback_reason="agent_error")

    monkeypatch.setattr(agents, "assess", fake_assess)
    rep = asyncio.run(assess_country(_inventory(), "MY", mock=True))
    assert rep.method == "fallback" and rep.fallback_reason == "agent_error"
    assert "Simplified check: x" in rep.notes
    assert rep.model_dump()["method"] == "fallback"


def test_agent_run_retried_on_rate_limit(monkeypatch):
    calls = []

    async def flaky(agent, message):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("critic_MY: 429 RESOURCE_EXHAUSTED")
        return "[]"

    async def no_sleep(_):
        return None

    monkeypatch.setattr(agents, "run_agent_once", flaky)
    monkeypatch.setattr(agents.asyncio, "sleep", no_sleep)

    class A:
        name = "critic_MY"

    assert asyncio.run(agents.run_agent(A(), "m")) == "[]" and len(calls) == 3
    calls.clear()

    async def broken(agent, message):
        calls.append(1)
        raise RuntimeError("bad request")

    monkeypatch.setattr(agents, "run_agent_once", broken)
    with pytest.raises(RuntimeError):
        asyncio.run(agents.run_agent(A(), "m"))
    assert len(calls) == 1  # other errors are not retried


def test_hung_agent_times_out_and_falls_back(real_provisions, monkeypatch):
    from app import config

    async def hang(agent, message):
        await asyncio.sleep(60)

    inv = _inventory()
    monkeypatch.setattr(config, "AGENT_TIMEOUT_S", 0.05)
    monkeypatch.setattr(agents, "_run_agent", hang)
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **kw: [])
    res = asyncio.run(agents.assess(inv, "MY", mock=False))
    assert res.method == "fallback" and res.fallback_reason == "timeout"
    # the critic gives up too, and says so
    verdicts, notes = asyncio.run(agents.critique(agents.mock_assess(inv, "MY")[:1], inv, "MY", mock=False))
    assert verdicts == [] and notes and notes[0].startswith("Critic unavailable")


def test_empty_reply_reason(real_provisions, monkeypatch):
    async def empty(agent, message):
        return "[]"

    monkeypatch.setattr(agents, "run_agent", empty)
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **kw: [])
    res = asyncio.run(agents.assess(_inventory(), "MY", mock=False))
    assert res.fallback_reason == "empty_reply"
