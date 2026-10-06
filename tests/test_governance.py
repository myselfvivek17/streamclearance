"""Governance: each report records how it was made; the UI's privacy claims come from config."""
import hashlib

from fastapi.testclient import TestClient

from app import config, provenance, provisions
from app.main import app
from tests.test_privacy import _analyse, _upload


def test_report_records_provenance(tiny_video, real_provisions):
    with TestClient(app) as client:
        job = _analyse(client, _upload(client, tiny_video[0])["video_id"], ("MY", "PK"))
    assert job["status"] == "done"
    report = job["report"]
    assert report["disclaimer"] == "Automated pre-check, not legal advice. A person makes the publish decision."
    pv = report["provenance"]
    assert pv["created_at"].endswith("Z") and pv["app_git_sha"] and pv["brain"] == "adk"
    assert pv["model"] == {"backend": "mock", "fast": "mock", "smart": "mock", "location": None}

    my = {law["law_id"]: law for law in pv["laws"] if law["country"] == "MY"}
    osa = provisions.get_law("MY/OSA2025")
    assert set(my) <= {law.id for law in provisions.laws_for("MY")} and "OSA2025" in my
    assert my["OSA2025"]["title"] == osa.title and my["OSA2025"]["status"] in ("draft", "reviewed")
    assert my["OSA2025"]["source_sha256"] == hashlib.sha256(osa.pdf_path.read_bytes()).hexdigest()
    assert my["OSA2025"]["provisions"] == sum(p.law_id == "OSA2025" for p in provisions.provisions_for("MY"))
    assert my["OSA2025"]["provisions_built"]

    # PK has no provisions: its rule cards are the law version (drafts)
    pk = [law for law in pv["laws"] if law["country"] == "PK"]
    assert len(pk) == 1 and pk[0]["law_id"] == config.COUNTRIES["PK"].source and pk[0]["status"] == "draft"


def test_provenance_model_and_location(monkeypatch):
    monkeypatch.delenv("SC_MOCK")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setattr(config, "DATA_RESIDENCY", None)
    monkeypatch.setenv("GCP_VERTEX_LOCATION", "global")
    m = provenance.model_info(mock=False)
    assert m.backend == "gemini-vertex" and m.smart == config.SC_MODEL_SMART and m.location == "global"


def test_build_info_file_wins(tmp_path, monkeypatch):
    path = tmp_path / "build_info.json"
    path.write_text('{"git_sha": "abc1234", "file_dates": {}}')
    monkeypatch.setattr(provenance, "BUILD_INFO_PATH", path)
    provenance.build_info.cache_clear()
    try:
        assert provenance.provenance([], mock=True, created_at=0).app_git_sha == "abc1234"
    finally:
        provenance.build_info.cache_clear()


def test_status_privacy_claims(monkeypatch):
    monkeypatch.setenv("SC_REPORT_TTL_DAYS", "3")
    with TestClient(app) as client:
        pv = client.get("/api/status").json()["privacy"]
    assert pv == {"video_deleted_after_check": True, "report_ttl_days": 3, "no_training": True}
    # AI Studio (free tier may train): no such claim
    monkeypatch.delenv("SC_MOCK")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert config.no_training() is False


def test_privacy_page(monkeypatch):
    monkeypatch.setenv("SC_ACCESS_CODE", "secret")  # public: not behind the access code
    monkeypatch.setenv("SC_APP_REGION", "asia-southeast1")
    with TestClient(app) as client:
        r = client.get("/privacy")
        assert r.status_code == 200
        page = r.text
        assert "{{" not in page and "7 days" in page and "60 minutes" in page
        assert "Singapore" in page and "[CONTACT EMAIL]" in page and "Google Cloud" in page
        assert "never used to train" in page
        assert "Malaysia PDPA 2010 and India DPDP Act 2023 principles in mind" in page
        assert "compliant" not in page.lower()
        assert 'href="/privacy"' in client.get("/").text  # footer link


def test_privacy_page_vertex_global(monkeypatch):
    monkeypatch.delenv("SC_MOCK")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setattr(config, "DATA_RESIDENCY", None)
    monkeypatch.setenv("GCP_VERTEX_LOCATION", "global")
    with TestClient(app) as client:
        page = client.get("/privacy").text
    assert "no data residency" in page and "up to 24 hours" in page
