"""Temporary OpenRouter bridge (delete with app/openrouter_backend.py). No network: httpx is mocked."""
import base64
import json
import logging

import httpx
import pytest
from pydantic import BaseModel

from app import config, gemini, pipeline
from app import openrouter_backend as ob

FAKE_KEY = "sk-or-v1-FAKEKEY-must-never-be-logged-0123456789"


class Tiny(BaseModel):
    answer: str
    n: int


def ok(content: str, cost: float = 0) -> httpx.Response:
    return httpx.Response(200, json={
        "model": "x", "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": cost}})


def err(code: int, message: str = "nope", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(code, json={"error": {"code": code, "message": message}}, headers=headers)


@pytest.fixture
def router(monkeypatch, tmp_path):
    """Install a fake OpenRouter. Usage: calls = router(handler); handler(body, n) -> Response."""
    for var in ("SC_MOCK", "SC_PROVIDER", "GEMINI_API_KEY", "SC_OR_MODEL", "SC_OR_AUDIO_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    sleeps: list[float] = []
    monkeypatch.setattr(ob, "_sleep", sleeps.append)
    monkeypatch.setattr(gemini, "_sleep", sleeps.append)
    monkeypatch.setattr(ob, "USAGE_LOG", tmp_path / "usage.log")
    monkeypatch.setattr(ob, "_halted", None)
    monkeypatch.setattr(ob, "MIN_INTERVAL_S", 0)

    def install(handler):
        calls: list[tuple[httpx.Request, dict]] = []

        def h(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append((request, body))
            return handler(body, len(calls))

        monkeypatch.setattr(ob, "_transport", httpx.MockTransport(h))
        return calls

    install.sleeps = sleeps
    return install


@pytest.mark.parametrize("env, expected", [
    ({}, "mock"),
    ({"OPENROUTER_API_KEY": "k"}, "openrouter"),
    ({"GEMINI_API_KEY": "g", "OPENROUTER_API_KEY": "k"}, "gemini"),
    ({"GEMINI_API_KEY": "g", "OPENROUTER_API_KEY": "k", "SC_PROVIDER": "openrouter"}, "openrouter"),
    ({"OPENROUTER_API_KEY": "k", "SC_PROVIDER": "mock"}, "mock"),
    ({"SC_PROVIDER": "gemini"}, "gemini"),
    ({"OPENROUTER_API_KEY": "k", "SC_PROVIDER": "openrouter", "SC_MOCK": "1"}, "mock"),
])
def test_dispatch_order(monkeypatch, env, expected):
    for var in ("SC_MOCK", "SC_PROVIDER", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert config.provider() == expected
    assert config.mock_mode() == (expected == "mock")


def test_bad_provider_rejected(monkeypatch):
    monkeypatch.delenv("SC_MOCK", raising=False)
    monkeypatch.setenv("SC_PROVIDER", "vertex")
    with pytest.raises(ValueError):
        config.provider()


def test_json_schema_request_shape(router):
    calls = router(lambda body, n: ok('{"answer": "yes", "n": 2}'))
    out = gemini.generate_json("Say yes.", Tiny, model="gemini-3.8-flash", system_instruction="Be terse.")
    assert out == {"answer": "yes", "n": 2}
    assert len(calls) == 1
    req, body = calls[0]
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert req.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert req.headers["X-Title"] == "StreamClearance"
    assert body["model"] == ob.DEFAULT_MODEL and body["model"].endswith(":free")
    assert body["usage"] == {"include": True}
    assert "provider" not in body
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert set(rf["json_schema"]["schema"]["properties"]) == {"answer", "n"}
    assert body["messages"][0] == {"role": "system", "content": "Be terse."}
    assert body["messages"][1]["content"] == [{"type": "text", "text": "Say yes."}]


def test_fallback_to_json_object_then_prompt(router):
    def handler(body, n):
        if n == 1:
            return err(400, "response_format json_schema is not supported")
        if n == 2:
            return ok("not json at all")
        return ok('Sure!\n```json\n{"items": [{"answer": "a", "n": 1}]}\n```')

    calls = router(handler)
    out = ob.generate_json("List.", list[Tiny])
    assert out == [{"answer": "a", "n": 1}]
    modes = [(b.get("response_format") or {}).get("type") for _, b in calls]
    assert modes == ["json_schema", "json_object", None]
    hint = calls[1][1]["messages"][-1]["content"][-1]["text"]
    assert "JSON Schema" in hint and '"items"' in hint


def test_repair_retry(router):
    replies = iter([ok('{"answer": "x"}')] * 3 + [ok('{"answer": "x", "n": 0}')])
    calls = router(lambda body, n: next(replies))
    assert ob.generate_json("q", Tiny) == {"answer": "x", "n": 0}
    assert len(calls) == 4 and "Fix this JSON" in calls[-1][1]["messages"][-1]["content"][0]["text"]


def test_model_fallback_and_retry_after(router):
    def handler(body, n):
        if body["model"] == ob.DEFAULT_MODEL:
            return err(429, "rate limited", headers={"Retry-After": "7"})
        return ok('{"answer": "b", "n": 3}')

    calls = router(handler)
    assert ob.generate_json("q", Tiny) == {"answer": "b", "n": 3}
    assert [b["model"] for _, b in calls] == [ob.DEFAULT_MODEL, ob.FALLBACK_MODEL]
    assert 7 in router.sleeps


def test_refuses_paid_model(router, monkeypatch):
    calls = router(lambda body, n: ok('{"answer": "x", "n": 1}'))
    monkeypatch.setenv("SC_OR_MODEL", "google/gemini-3.8-flash")
    with pytest.raises(ob.PaidModelRefused):
        ob.generate_json("q", Tiny)
    monkeypatch.setenv("SC_OR_MODEL", ob.DEFAULT_MODEL)
    monkeypatch.setenv("SC_OR_AUDIO_MODEL", "openai/gpt-audio")
    with pytest.raises(ob.PaidModelRefused):
        ob.audio_model()
    assert calls == []


def test_nonzero_cost_halts(router):
    calls = router(lambda body, n: ok('{"answer": "x", "n": 1}', cost=0.0001))
    with pytest.raises(ob.NonZeroCost):
        ob.generate_json("q", Tiny)
    with pytest.raises(ob.NonZeroCost):
        ob.generate_json("q", Tiny)
    assert len(calls) == 1
    assert '"cost": 0.0001' in ob.USAGE_LOG.read_text()


def test_video_part_shape(router, tiny_video):
    video, real = tiny_video
    inventory = json.dumps({"languages": ["en"], "segments": [{"start_s": 0, "end_s": 2}]})

    def handler(body, n):
        if body["model"] == ob.DEFAULT_AUDIO_MODEL:
            return ok("[0.0-1.5] Hello there.")
        return ok(inventory)

    calls = router(handler)
    out = gemini.generate_json_with_video(video, "video/mp4", "Inventory please.", pipeline.StageADraft)
    video_calls = [b for _, b in calls if b["model"] != ob.DEFAULT_AUDIO_MODEL]
    parts = video_calls[0]["messages"][-1]["content"]
    assert parts[0]["type"] == "video_url"
    url = parts[0]["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(url.split(",", 1)[1])
    assert parts[1]["type"] == "text" and parts[1]["text"].startswith("Inventory please.")
    if real:  # ffmpeg present: audio was extracted and transcribed by the free audio model
        audio = [b for _, b in calls if b["model"] == ob.DEFAULT_AUDIO_MODEL][0]
        apart = audio["messages"][0]["content"][1]
        assert apart["type"] == "input_audio" and apart["input_audio"]["format"] == "mp3"
        assert "[0.0-1.5] Hello there." in parts[1]["text"]
        assert out["speech_transcribed"] is True
    else:
        assert out["speech_transcribed"] is False


def test_key_and_payload_never_logged(router, tiny_video, caplog):
    caplog.set_level(logging.DEBUG)
    video, _ = tiny_video

    def handler(body, n):
        if n == 1:
            return err(400, "json_schema unsupported")
        if n == 2:
            return err(503, "upstream down")
        return ok('{"answer": "x", "n": 1}')

    router(handler)
    ob.generate_json("q", Tiny)
    with pytest.raises(Exception) as excinfo:
        router(lambda body, n: err(401, "bad key"))
        ob.generate_json_with_video(video, "video/mp4", "p", pipeline.StageADraft)
    texts = [caplog.text, ob.USAGE_LOG.read_text(), str(excinfo.value)]
    for t in texts:
        assert FAKE_KEY not in t
        assert "base64," not in t


def test_stage_a_carries_speech_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(gemini, "generate_json_with_video", lambda *a, **k: {
        "languages": [], "segments": [{"start_s": 0, "end_s": 1}], "speech_transcribed": False})
    inv = pipeline.stage_a(tmp_path / "v.mp4", 1.0, mock=False)
    assert inv.speech_transcribed is False
    assert pipeline.stage_a(tmp_path / "v.mp4", 1.0, mock=True).speech_transcribed is None
