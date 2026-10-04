import pytest

from app import gemini, pipeline
from app.schemas import Inventory


@pytest.mark.parametrize("duration", [2.0, 45.0, 600.0, None])
def test_mock_inventory_valid_and_scaled(duration):
    inv = pipeline.stage_a(pipeline.Path("x.mp4"), duration, mock=True)
    Inventory.model_validate(inv.model_dump())
    expected = duration or pipeline.MOCK_UNKNOWN_DURATION_S
    assert inv.duration_s == pytest.approx(expected)
    assert inv.segments[0].start_s == 0 and inv.segments[-1].end_s == pytest.approx(expected)
    for a, b in zip(inv.segments, inv.segments[1:]):
        assert a.end_s == pytest.approx(b.start_s)
    assert any(s.categories for s in inv.segments), "mock must contain flaggable content"
    assert pipeline.stage_a(pipeline.Path("x.mp4"), duration, mock=True) == inv  # deterministic


def test_real_stage_a_clamps_to_duration(monkeypatch, tmp_path):
    seen = {}

    def fake(path, mime, prompt, schema, **kw):
        seen.update(mime=mime, prompt=prompt, schema=schema)
        return {"languages": ["en"], "segments": [
            {"start_s": 0, "end_s": 4, "transcript": "hi", "categories": ["other"]},
            {"start_s": 4, "end_s": 12, "transcript": "bye"},
            {"start_s": 30, "end_s": 35, "transcript": "hallucinated"},
        ]}

    monkeypatch.setattr(gemini, "generate_json_with_video", fake)
    inv = pipeline.stage_a(tmp_path / "v.webm", 10.0, mock=False)
    assert seen["mime"] == "video/webm" and "10.0 seconds" in seen["prompt"]
    assert [s.end_s for s in inv.segments] == [4, 10.0]


def test_retry_on_429(monkeypatch):
    monkeypatch.setattr(gemini, "_sleep", lambda s: None)
    calls = []

    class RateLimited(Exception):
        code = 429

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RateLimited("RESOURCE_EXHAUSTED")
        return "ok"

    assert gemini.with_retry(flaky, what="t", retries=5) == "ok" and len(calls) == 3

    class BadRequest(Exception):
        code = 400

    def bad():
        raise BadRequest("nope")

    with pytest.raises(BadRequest):
        gemini.with_retry(bad, what="t")
