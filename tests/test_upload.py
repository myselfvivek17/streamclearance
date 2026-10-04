import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app

client = TestClient(app)


def _upload(path, name="clip.mp4", ctype="video/mp4"):
    with open(path, "rb") as fh:
        return client.post("/api/upload", files={"file": (name, fh, ctype)})


def test_upload_ok_and_served(tiny_video):
    path, real = tiny_video
    r = _upload(path)
    assert r.status_code == 200, r.text
    meta = r.json()
    assert len(meta["video_id"]) == 32 and meta["mime_type"] == "video/mp4"
    if real:
        assert 1.5 < meta["duration_s"] < 2.5
    v = client.get(f"/api/videos/{meta['video_id']}", headers={"Range": "bytes=0-99"})
    assert v.status_code == 206 and len(v.content) == 100


@pytest.mark.parametrize("name,ctype", [("clip.avi", "video/x-msvideo"), ("clip.txt", "text/plain"),
                                        ("clip.mp4", "text/plain"), ("noext", "video/mp4")])
def test_upload_rejects_wrong_type(tiny_video, name, ctype):
    assert _upload(tiny_video[0], name, ctype).status_code == 415


def test_upload_rejects_too_large(tiny_video, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    r = _upload(tiny_video[0])
    assert r.status_code == 413
    from app.storage import get_storage
    assert not any(get_storage().uploads.glob("*.mp4"))


def test_upload_rejects_too_long(tiny_video, monkeypatch):
    path, real = tiny_video
    if not real:
        pytest.skip("ffmpeg/ffprobe not available")
    monkeypatch.setattr(config, "MAX_DURATION_S", 1.0)
    r = _upload(path)
    assert r.status_code == 400 and "limit" in r.json()["detail"]


def test_upload_rejects_unreadable_video(tmp_path):
    if not config.ffprobe_path():
        pytest.skip("ffprobe not available")
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not a video" * 100)
    assert _upload(fake).status_code == 400


def test_unknown_video_404():
    assert client.get("/api/videos/" + "0" * 32).status_code == 404
    assert client.get("/api/videos/..%2Fsecret").status_code == 404
