"""Privacy: the video is deleted when its check ends; unused uploads expire."""
import asyncio
import os
import time

import pytest
from fastapi.testclient import TestClient

from app import config, gemini, jobs, retention
from app.main import app
from app.storage import get_storage


def _upload(client, path):
    with open(path, "rb") as fh:
        return client.post("/api/upload", files={"file": ("my holiday.mp4", fh, "video/mp4")}).json()


def _wait(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _files_for(video_id):
    return sorted(p.name for p in get_storage().uploads.glob(f"{video_id}*"))


def _analyse(client, video_id, countries=("MY",)):
    r = client.post("/api/analyze", json={"video_id": video_id, "countries": list(countries)})
    assert r.status_code == 200, r.text
    return _wait(client, r.json()["job_id"])


def test_video_deleted_after_done(tiny_video):
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0])["video_id"]
        # a derived file stored under the video id goes too
        (get_storage().uploads / f"{video_id}.frames").mkdir()
        assert len(_files_for(video_id)) == 3
        job = _analyse(client, video_id)
        assert job["status"] == "done"
        assert _files_for(video_id) == []
        assert job["video_deleted_at"].endswith("Z")
        privacy = job["report"]["privacy"]
        assert privacy["video_deleted_at"] == job["video_deleted_at"]
        assert privacy["report_expires_at"] == job["expires_at"]
        # the report needs no video: it can't be fetched any more, the report still can
        assert client.get(f"/api/videos/{video_id}").status_code == 404
        assert client.get(f"/api/jobs/{job['job_id']}").json()["report"]["inventory"]["segments"]
        # and it can't be analysed twice
        r = client.post("/api/analyze", json={"video_id": video_id, "countries": ["MY"]})
        assert r.status_code == 404


def test_video_deleted_after_failure(tiny_video, monkeypatch):
    from app import pipeline

    def boom(*a, **kw):
        raise RuntimeError("inventory exploded")

    monkeypatch.setenv("SC_BRAIN", "legacy")
    monkeypatch.setattr(pipeline, "stage_a", boom)
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0])["video_id"]
        job = _analyse(client, video_id)
        assert job["status"] == "error" and job["video_deleted_at"]
        assert _files_for(video_id) == []


def test_video_deleted_after_cancel(tiny_video, monkeypatch):
    monkeypatch.setenv("SC_MOCK_DELAY", "5")
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0])["video_id"]

    async def go():
        job = jobs.create_job(video_id, ["MY"])
        jobs.start_job(job)
        assert video_id in jobs.active_video_ids()
        await asyncio.sleep(0.2)
        task, _ = jobs._active[job["job_id"]]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return job["job_id"]

    job_id = asyncio.run(go())
    assert _files_for(video_id) == []
    saved = get_storage().load_job(job_id)
    assert saved["status"] == "cancelled" and saved["video_deleted_at"]
    assert video_id not in jobs.active_video_ids()


def test_unused_video_swept(tiny_video, monkeypatch):
    with TestClient(app) as client:
        old = _upload(client, tiny_video[0])["video_id"]
        busy = _upload(client, tiny_video[0])["video_id"]
        fresh = _upload(client, tiny_video[0])["video_id"]
    long_ago = time.time() - 61 * 60
    for vid in (old, busy):
        for p in get_storage().uploads.glob(f"{vid}*"):
            os.utime(p, (long_ago, long_ago))
    assert retention.sweep_unused_videos(active={busy}) == 1
    assert _files_for(old) == [] and _files_for(busy) and _files_for(fresh)
    monkeypatch.setenv("SC_UNUSED_VIDEO_TTL_MIN", "0")
    assert retention.sweep_unused_videos(active=set()) == 2


def test_gcs_copy_deleted(tiny_video, monkeypatch):
    """On Vertex, a video too large to inline has an upload-bucket object: purge deletes it."""
    monkeypatch.delenv("SC_MOCK")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setattr(config, "VERTEX_INLINE_MAX_BYTES", 10)
    calls = []

    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._body = code, body or {}

        def json(self):
            return self._body

    class Session:
        def get(self, url, params, timeout):
            calls.append(("list", params["prefix"]))
            return Resp(200, {"items": [{"name": params["prefix"] + "abc.mp4"}]})

        def delete(self, url, timeout):
            calls.append(("delete", url.rsplit("/", 1)[-1]))
            return Resp(204)

    monkeypatch.setattr(gemini, "_gcs_session", lambda: Session())
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0])["video_id"]
    retention.purge_video(video_id)
    assert _files_for(video_id) == []
    assert calls == [("list", f"videos/{video_id}/"), ("delete", f"videos%2F{video_id}%2Fabc.mp4")]


# ---------------------------------------------------------------- report retention + delete now

def _age(job_id, days):
    store = get_storage()
    job = store.load_job(job_id)
    job["created_at"] -= days * 86400
    store.save_job(job_id, job)


def test_expired_report_deleted_on_read(tiny_video, monkeypatch):
    monkeypatch.setenv("SC_REPORT_TTL_DAYS", "7")
    with TestClient(app) as client:
        job = _analyse(client, _upload(client, tiny_video[0])["video_id"])
        job_id = job["job_id"]
        _age(job_id, 6.9)
        assert client.get(f"/api/jobs/{job_id}").status_code == 200
        _age(job_id, 0.2)
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 404 and r.json()["detail"] == "This report expired and was deleted"
        assert get_storage().load_job(job_id) is None
        assert client.get(f"/api/jobs/{job_id}").json()["detail"] == "This report expired and was deleted"


def test_sweeper_expires_reports(tiny_video, monkeypatch):
    from app import main

    with TestClient(app) as client:
        old = _analyse(client, _upload(client, tiny_video[0])["video_id"])["job_id"]
        new = _analyse(client, _upload(client, tiny_video[0])["video_id"])["job_id"]
    _age(old, 8)
    main.sweep()
    store = get_storage()
    assert store.load_job(old) is None and store.gone_reason(old) == "expired"
    assert store.load_job(new) is not None
    # the content-free marker itself goes after another TTL
    marker = store.jobs / "gone" / old
    assert marker.read_text() == "expired"
    t = time.time() - 8 * 86400
    os.utime(marker, (t, t))
    main.sweep()
    assert not marker.exists()


def test_delete_report_now(tiny_video):
    with TestClient(app) as client:
        job_id = _analyse(client, _upload(client, tiny_video[0])["video_id"])["job_id"]
        r = client.delete(f"/api/jobs/{job_id}")
        assert r.status_code == 200 and r.json() == {"deleted": job_id}
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 404 and r.json()["detail"] == "This report was deleted"
        assert not (get_storage().jobs / f"{job_id}.json").exists()
        assert client.delete(f"/api/jobs/{job_id}").status_code == 404
        assert client.delete("/api/jobs/" + "b" * 32).status_code == 404


def test_delete_running_job(tiny_video, monkeypatch):
    monkeypatch.setenv("SC_MOCK_DELAY", "5")
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0])["video_id"]
        job_id = client.post("/api/analyze", json={"video_id": video_id, "countries": ["MY"]}).json()["job_id"]
        assert client.delete(f"/api/jobs/{job_id}").status_code == 200
        assert _files_for(video_id) == []
        time.sleep(0.2)
        assert get_storage().load_job(job_id) is None  # not written back by the cancelled task
        assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_delete_needs_access_code(tiny_video, monkeypatch):
    with TestClient(app) as client:
        job_id = _analyse(client, _upload(client, tiny_video[0])["video_id"])["job_id"]
        video_id = _upload(client, tiny_video[0])["video_id"]
        monkeypatch.setenv("SC_ACCESS_CODE", "letmein")
        assert client.delete(f"/api/jobs/{job_id}").status_code == 401
        assert client.delete(f"/api/videos/{video_id}").status_code == 401
        assert get_storage().load_job(job_id) is not None
        h = {"X-Access-Code": "letmein"}
        assert client.delete(f"/api/videos/{video_id}", headers=h).status_code == 200
        assert _files_for(video_id) == []
        assert client.delete(f"/api/jobs/{job_id}", headers=h).status_code == 200
