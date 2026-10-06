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


def test_parse_list_skips_prose_id_lists():
    from app.brain.agents import MomentAssessment, _parse_list

    text = ('Relevant provisions ["IN/DMR1954/s3", "IN/DMR1954/Sch", "IN/DMR1954/s4"].\n'
            '```json\n[{"start_s": 1, "end_s": 2, "leaf": "health.cure_claim"}]\n```')
    out = _parse_list(text, MomentAssessment)
    assert [a.leaf for a in out] == ["health.cure_claim"]


class _SlowFirst:
    """First request hangs for `first_s`, later ones answer at once."""

    def __init__(self, first_s):
        import threading
        self.n, self.first_s, self.lock = 0, first_s, threading.Lock()
        self.models = self

    def generate_content(self, model, contents, config):
        import time
        with self.lock:
            self.n += 1
            n = self.n
        if n == 1:
            time.sleep(self.first_s)
        return SimpleNamespace(text=json.dumps({"n": n}), model_version=model, usage_metadata=None)


def test_slow_call_is_hedged_and_the_first_answer_wins(tmp_path, monkeypatch):
    from app import metrics

    monkeypatch.setattr(gemini, "USAGE_LOG", tmp_path / "usage.log")
    client = _SlowFirst(first_s=1.0)
    budget = gemini.job_budget("j", limit=5)
    m = metrics.start("j")
    with metrics.stage("translation:MY"):
        resp = gemini._generate(client, "m", "p", None, what="generate_json", hedge_after=0.05)
    assert json.loads(resp.text) == {"n": 2} and budget.used == 2
    assert m.snapshot()[1]["translation:MY"]["hedges"] == 1


def test_no_hedge_when_fast_or_out_of_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(gemini, "USAGE_LOG", tmp_path / "usage.log")
    budget = gemini.job_budget("j", limit=5)
    resp = gemini._generate(_SlowFirst(first_s=0), "m", "p", None, what="x", hedge_after=1.0)
    assert json.loads(resp.text) == {"n": 1} and budget.used == 1
    budget = gemini.job_budget("j", limit=1)  # the duplicate would exceed the budget
    resp = gemini._generate(_SlowFirst(first_s=0.3), "m", "p", None, what="x", hedge_after=0.05)
    assert json.loads(resp.text) == {"n": 1} and budget.used == 1


def test_video_hedge_scales_with_duration(monkeypatch):
    assert config.video_hedge_after(26) == pytest.approx(15 + 13)
    monkeypatch.setattr(config, "HEDGE_AFTER_S", 0)
    assert config.video_hedge_after(26) is None
