"""Moment thumbnails (app/frames.py): taken before the video is deleted, served behind
the access gate, deleted with the report."""
import time

import pytest
from fastapi.testclient import TestClient

from app import config, frames, main
from app.main import app
from app.storage import get_storage


def _analyse(client, path):
    with open(path, "rb") as fh:
        vid = client.post("/api/upload", files={"file": ("v.mp4", fh, "video/mp4")}).json()["video_id"]
    job_id = client.post("/api/analyze", json={"video_id": vid, "countries": ["MY"]}).json()["job_id"]
    deadline = time.time() + 30
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture
def real_video(tiny_video):
    path, real = tiny_video
    if not real or not config.ffmpeg_path():
        pytest.skip("needs ffmpeg")
    return path


def _framed(job):
    return [s for s in job["report"]["inventory"]["segments"] if s.get("frame")]


def test_frames_taken_and_served(real_video, monkeypatch):
    with TestClient(app) as client:
        job = _analyse(client, real_video)
        assert job["status"] == "done"
        segs = job["report"]["inventory"]["segments"]
        framed = _framed(job)
        # one frame per segment stage A tagged (the mock tags 2 of 3), none for neutral ones
        assert len(framed) == sum(1 for s in segs if s["leaves"]) == 2
        assert all(s["leaves"] for s in framed)
        url = f"/api/jobs/{job['job_id']}/frames/{framed[0]['frame']}"
        r = client.get(url)
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"  # JPEG
        assert any(e["kind"] == "watch" and "still frame" in e["text"] for e in job["events"])
        # the video itself is gone; the frames stay with the report
        assert get_storage().video_path(job["video_id"]) is None
        # gated like every /api route, and no path tricks
        monkeypatch.setenv("SC_ACCESS_CODE", "letmein")
        assert client.get(url).status_code == 401
        monkeypatch.delenv("SC_ACCESS_CODE")
        assert client.get(f"/api/jobs/{job['job_id']}/frames/..%2F{job['job_id']}.json").status_code == 404
        assert client.get(f"/api/jobs/{job['job_id']}/frames/x.png").status_code == 404


def test_frames_deleted_with_report(real_video):
    with TestClient(app) as client:
        job = _analyse(client, real_video)
        folder = get_storage().frames_dir(job["job_id"])
        assert any(folder.iterdir())
        assert client.delete(f"/api/jobs/{job['job_id']}").status_code == 200
        assert not folder.parent.exists()
        assert client.get(f"/api/jobs/{job['job_id']}/frames/{_framed(job)[0]['frame']}").status_code == 404


def test_frames_deleted_when_report_expires(real_video):
    with TestClient(app) as client:
        job = _analyse(client, real_video)
    store = get_storage()
    rec = store.load_job(job["job_id"])
    rec["created_at"] -= 8 * 86400
    store.save_job(job["job_id"], rec)
    main.sweep()
    assert not store.frames_dir(job["job_id"]).parent.exists()


def test_orphan_frame_folders_swept():
    store = get_storage()
    orphan = store.frames_dir("c" * 32)
    orphan.mkdir(parents=True)
    (orphan / "0.jpg").write_bytes(b"x")
    main.sweep()
    assert not orphan.parent.exists()


def test_no_ffmpeg_no_frames_no_error(tiny_video, monkeypatch):
    monkeypatch.setattr(config, "ffmpeg_path", lambda: None)
    with TestClient(app) as client:
        job = _analyse(client, tiny_video[0])
    assert job["status"] == "done" and not _framed(job)


def test_frame_names():
    assert frames.frame_name(12.5) == "12500.jpg" and frames.valid_name("12500.jpg")
    assert not frames.valid_name("../x.jpg") and not frames.valid_name("1.jpeg")


def test_privacy_page_mentions_frames(monkeypatch):
    with TestClient(app) as client:
        monkeypatch.setattr(config, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
        assert "deleted with it" in client.get("/privacy").text
        monkeypatch.setattr(config, "ffmpeg_path", lambda: None)
        assert "keeps no images from your video" in client.get("/privacy").text
