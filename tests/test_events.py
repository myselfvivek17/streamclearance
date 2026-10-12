"""Agent activity feed (app/events.py): real pipeline steps, in order, stored with the job."""
import time

from fastapi.testclient import TestClient

from app import events
from app.main import app


def _run(client, path, countries):
    with open(path, "rb") as fh:
        vid = client.post("/api/upload", files={"file": ("v.mp4", fh, "video/mp4")}).json()["video_id"]
    job_id = client.post("/api/analyze", json={"video_id": vid, "countries": countries}).json()["job_id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_mock_job_events_in_order(tiny_video, real_provisions):
    with TestClient(app) as client:
        job = _run(client, tiny_video[0], ["MY", "AU", "PK"])
        assert job["status"] == "done"
        evs = job["events"]
        assert evs and job["events_total"] == len(evs)
        assert all(e["kind"] in events.KINDS and e["text"] for e in evs)
        assert [e["t"] for e in evs] == sorted(e["t"] for e in evs)
        stages = [e["stage"] for e in evs]
        # stage A comes first, the report last; every country reports back
        assert stages[0] == "inventory" and stages[-1] == "report"
        assert any(e["kind"] == "watch" and e["text"].startswith("Watched ") for e in evs)
        first_country = stages.index("country")
        assert all(s == "inventory" for s in stages[:first_country] if s not in ("translation", "ai"))
        for cc in ("MY", "AU"):
            mine = [e for e in evs if e.get("country") == cc]
            assert any(e["kind"] == "law" and e["text"].startswith("Assessed ") for e in mine)
            assert any(e["kind"] == "review" for e in mine)
        assert any(e.get("country") == "PK" and e["kind"] == "warn" and "not checked" in e["text"]
                   for e in evs)
        assert any(e["kind"] == "translate" or (e["kind"] == "warn" and e["stage"] == "translation")
                   for e in evs)
        ready = next(e for e in evs if e["text"].startswith("Report ready"))
        assert "2 of 3 countries checked" in ready["text"]  # PK has no verified law
        assert evs[-1]["text"].startswith("Timings: total ")

        tail = client.get(f"/api/jobs/{job['job_id']}?since=3").json()
        assert tail["events"] == evs[3:] and tail["events_total"] == len(evs)


def test_feed_is_capped():
    log = events.EventLog("x", t0=time.time())
    for i in range(events.MAX_EVENTS + 50):
        log.add("info", f"step {i}", stage="country")
    evs = log.snapshot()
    assert len(evs) == events.MAX_EVENTS
    assert evs[-1]["text"].startswith("51 more step(s) not shown")


def test_emit_outside_a_job_is_a_no_op():
    events._log.set(None)
    events.emit("info", "nothing listens", stage="report")  # must not raise
    try:
        events.EventLog("x").add("nope", "bad kind", stage="report")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kind accepted")
