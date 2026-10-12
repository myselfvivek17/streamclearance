"""Logs carry no content: a canary phrase from the video's transcript must not
appear in any log line or in jobs/usage.log, whatever path the job takes."""
import json
import logging
import time

import pytest
from fastapi.testclient import TestClient

from app import gemini, logsafe, pipeline
from app.main import app
from app.pipeline import StageADraft

CANARY = "purple-walrus-7731 sings at midnight"


def _upload(client, path, name):
    with open(path, "rb") as fh:
        return client.post("/api/upload", files={"file": (name, fh, "video/mp4")}).json()


def _run(client, video_id, countries):
    job_id = client.post("/api/analyze", json={"video_id": video_id, "countries": countries}).json()["job_id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


class _Usage:
    prompt_token_count, candidates_token_count, thoughts_token_count, total_token_count = 11, 22, 0, 33


class _Resp:
    model_version = "fake-model-001"
    usage_metadata = _Usage()

    def __init__(self, payload):
        self.text = json.dumps(payload)


class _FakeModels:
    """Answers each pipeline call with output that quotes the canary, and fails
    Australia's call with an exception whose message quotes it too."""

    def generate_content(self, model, contents, config):
        schema = config.response_schema
        prompt = contents if isinstance(contents, str) else contents[-1]
        if schema is StageADraft:
            return _Resp({"languages": ["en"], "segments": [
                {"start_s": 0, "end_s": 1, "transcript": f"I promise {CANARY}, this cures diabetes",
                 "claims": [CANARY], "visuals": CANARY, "on_screen_text": CANARY,
                 "leaves": ["health.cure_claim", f"not.a.leaf {CANARY}"]},
                {"start_s": 1.5, "end_s": 1, "transcript": CANARY},  # end before start: dropped
            ]})
        if "Country: Australia" in prompt:
            raise RuntimeError(f"model said: {CANARY}")
        if prompt.startswith("Country:"):  # stage B: cite the first rule card
            cards = json.loads(prompt.split("RULE CARDS:\n", 1)[1].split("\n\nVIDEO INVENTORY", 1)[0])
            card = [c for c in cards if c["category"] != "other"][0]
            return _Resp([
                {"start_s": 0, "end_s": 1, "category": card["category"], "rule_card_ids": [card["id"]],
                 "risk": "red", "rationale": CANARY, "suggested_fix": CANARY},
                {"start_s": 0, "end_s": 1, "category": card["category"], "rule_card_ids": [f"{CANARY}"],
                 "risk": "red", "rationale": CANARY, "suggested_fix": CANARY},
                {"start_s": "soon " + CANARY, "rationale": CANARY},
            ])
        items = json.loads(prompt.split("\n", 1)[1])
        if "translation" in items[0]:  # judge
            return _Resp([{"index": x["index"], "drift": "increased", "explanation": CANARY} for x in items])
        if "languages listed for it" in prompt:  # translate (Malaysia: Malay)
            return _Resp([{"index": x["index"], "language": "Malay", "text": f"{x['text']} ({CANARY})"}
                          for x in items])
        return _Resp([{"index": x["index"], "text": f"{x['text']} ({CANARY})"} for x in items])


class _FakeClient:
    models = _FakeModels()


@pytest.fixture
def capture(caplog, tmp_path, monkeypatch):
    monkeypatch.setattr(gemini, "USAGE_LOG", tmp_path / "usage.log")
    caplog.set_level(logging.DEBUG)  # everything, including library debug lines
    return caplog


def _assert_clean(caplog, usage_log):
    text = "\n".join(r.getMessage() for r in caplog.records) + caplog.text
    assert CANARY not in text
    assert "purple-walrus" not in text and "diabetes" not in text
    assert "secret-name" not in text  # the user's file name is never logged either
    if usage_log.exists():
        assert CANARY not in usage_log.read_text()


def test_no_content_in_logs_real_pipeline_with_fake_model(tiny_video, capture, monkeypatch):
    """Legacy brain on the Gemini code path (fake client): stage A, stage B, the
    validator, translation drift, a failing country, the usage log."""
    monkeypatch.delenv("SC_MOCK")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setenv("SC_BRAIN", "legacy")
    monkeypatch.setenv("SC_LAW_SEARCH", "bm25")
    monkeypatch.setattr(gemini, "_get_client", lambda: _FakeClient())
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0], "secret-name.mp4")["video_id"]
        job = _run(client, video_id, ["MY", "AU"])
    assert job["status"] == "done" and not job["mock"], job.get("error")
    report = json.dumps(job["report"])
    assert CANARY in report  # the report holds the content; the logs must not
    assert CANARY in json.dumps(job["events"])  # so does the activity feed (report data)
    assert any("AU" in e and CANARY in e for e in job["report"]["errors"])
    my = next(c for c in job["report"]["countries"] if c["country"] == "MY")
    assert my["findings"] and my["translation"]
    usage = gemini.USAGE_LOG.read_text().splitlines()
    assert len(usage) >= 4 and all(json.loads(line)["total"] == 33 for line in usage)
    assert any("gemini usage" in r.getMessage() for r in capture.records)
    _assert_clean(capture, gemini.USAGE_LOG)


def test_no_content_in_logs_adk_mock(tiny_video, capture, real_provisions, monkeypatch):
    """ADK brain (mock) with the canary in the transcript, plus a failing job."""
    segs = [dict(s) for s in pipeline._MOCK_SEGMENTS]
    segs[1]["transcript"] = f"{CANARY} {segs[1]['transcript']}"
    monkeypatch.setattr(pipeline, "_MOCK_SEGMENTS", segs)
    with TestClient(app) as client:
        job = _run(client, _upload(client, tiny_video[0], "secret-name.mp4")["video_id"], ["MY", "IN"])
        assert job["status"] == "done" and CANARY in json.dumps(job["report"])
        assert CANARY in json.dumps(job["events"])  # feed text is report data, never logged

        def boom(*a, **kw):
            raise ValueError(f"cannot parse {CANARY}")

        monkeypatch.setattr(pipeline, "stage_a", boom)
        job = _run(client, _upload(client, tiny_video[0], "secret-name.mp4")["video_id"], ["MY"])
        assert job["status"] == "error" and CANARY in job["error"]
    _assert_clean(capture, gemini.USAGE_LOG)


def test_library_records_are_reduced(caplog):
    logsafe.install()
    caplog.set_level(logging.DEBUG)
    lib = logging.getLogger("google_adk.google.adk.models.google_llm")
    lib.debug(f"LLM Request: {CANARY}")  # dynamic text, no args
    lib.info("Sending out request, model: %s", CANARY)  # template + args
    try:
        raise RuntimeError(CANARY)
    except RuntimeError:
        lib.exception("call failed")
        logging.getLogger("app.jobs").exception("job %s failed", "abc")
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs == [logsafe.WITHHELD, "Sending out request, model: %s",
                    f"{logsafe.WITHHELD} [RuntimeError]", "job abc failed [RuntimeError]"]
    assert CANARY not in caplog.text
