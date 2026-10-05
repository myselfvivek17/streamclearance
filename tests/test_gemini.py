"""Backend selection, the per-job call cap and usage logging (no network)."""
import json
from types import SimpleNamespace

import pytest

from app import config, gemini


def test_backend_selection(monkeypatch):
    assert config.gemini_backend() is None and config.provider() == "mock"
    monkeypatch.delenv("SC_MOCK")
    assert config.provider() == "mock"
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert config.gemini_backend() == "studio" and config.provider() == "gemini"
    monkeypatch.setenv("GCP_PROJECT", "p")
    assert config.gemini_backend() == "vertex" and config.provider() == "gemini"
    monkeypatch.setenv("SC_GEMINI_BACKEND", "studio")
    assert config.gemini_backend() == "studio"
    monkeypatch.setenv("SC_MOCK", "1")
    assert config.provider() == "mock"


def test_apply_adk_env(monkeypatch):
    # setenv first so monkeypatch restores what apply_adk_env writes
    for var in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.setenv(var, "unset")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    monkeypatch.delenv("GCP_VERTEX_LOCATION", raising=False)
    monkeypatch.setenv("GCP_PROJECT", "proj")
    config.apply_adk_env()
    import os
    assert os.environ["GOOGLE_GENAI_USE_VERTEXAI"] == "TRUE"
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "proj"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "global"


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents))
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=2,
                                thoughts_token_count=3, total_token_count=15)
        return SimpleNamespace(text='{"ok": true}', model_version=model, usage_metadata=usage)


@pytest.fixture
def fake_vertex(monkeypatch, tmp_path):
    monkeypatch.setenv("GCP_PROJECT", "proj")
    client = SimpleNamespace(models=_FakeModels())
    monkeypatch.setattr(gemini, "_client", client)
    monkeypatch.setattr(gemini, "_client_backend", "vertex")
    monkeypatch.setattr(gemini, "USAGE_LOG", tmp_path / "usage.log")
    return client


def test_usage_logged_and_cap_enforced(fake_vertex):
    import contextvars

    def job():
        gemini.job_budget("job1", limit=2)
        assert gemini.generate_json("a", dict) == {"ok": True}
        gemini.generate_json("b", dict)
        with pytest.raises(gemini.CallBudgetExceeded):
            gemini.generate_json("c", dict)

    contextvars.copy_context().run(job)
    assert len(fake_vertex.models.calls) == 2
    lines = [json.loads(x) for x in gemini.USAGE_LOG.read_text().splitlines()]
    assert [x["job"] for x in lines] == ["job1", "job1"]
    assert lines[0]["prompt"] == 10 and lines[0]["thoughts"] == 3 and lines[0]["total"] == 15


def test_vertex_video_inline(fake_vertex, tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00" * 100)
    assert gemini.generate_json_with_video(video, "video/mp4", "go", dict) == {"ok": True}
    (_, contents), = fake_vertex.models.calls
    assert contents[0].inline_data.data == b"\x00" * 100 and contents[1] == "go"


def test_extract_json():
    assert gemini.extract_json('Sure:\n```json\n[{"a": 1}]\n```') == [{"a": 1}]
    with pytest.raises(ValueError):
        gemini.extract_json("nothing here")
