import re
import time

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.schemas import Report


def _upload(client, path, **headers):
    with open(path, "rb") as fh:
        return client.post("/api/upload", files={"file": ("clip.mp4", fh, "video/mp4")},
                           headers=headers)


def _wait(client, job_id, timeout=20, **kw):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}", **kw).json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_countries_and_pdf():
    with TestClient(app) as client:
        cs = client.get("/api/countries").json()
        assert [c["code"] for c in cs] == list(config.COUNTRIES)
        sg = next(c for c in cs if c["code"] == "SG")
        assert sg["label"] is None and "Act 2025" in sg["law"]
        assert all(c["rules_status"] == "draft/unreviewed" for c in cs)
        pdf = client.get(cs[0]["pdf_url"])
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
        assert pdf.headers["content-type"] == "application/pdf"
        assert pdf.headers["content-disposition"].startswith("inline")
        assert client.get("/api/laws/XX/pdf").status_code == 404
        assert client.get("/api/status").json()["mock"] is True


def test_job_lifecycle_end_to_end(tiny_video):
    path, real = tiny_video
    with TestClient(app) as client:
        video_id = _upload(client, path).json()["video_id"]
        r = client.post("/api/analyze", json={"video_id": video_id, "countries": ["pk", "AU", "SG", "PK"]})
        assert r.status_code == 200, r.text
        job = _wait(client, r.json()["job_id"])
        assert job["status"] == "done", job
        assert job["mock"] is True and job["countries"] == ["PK", "AU", "SG"]
        keys = [s["key"] for s in job["stages"]]
        assert keys == ["uploaded", "inventory", "country:PK", "country:AU", "country:SG",
                        "translation", "done"]
        assert all(s["status"] == "done" for s in job["stages"])

        report = Report.model_validate(job["report"])
        assert report.mock and report.video_id == video_id and not report.errors
        if real:
            assert report.duration_s == pytest.approx(2.0, abs=0.3)
        assert [c.country for c in report.countries] == ["PK", "AU", "SG"]
        for c in report.countries:
            assert all(f.end_s <= report.duration_s for f in c.findings)
            assert all(cid.startswith(c.country) for f in c.findings for cid in f.rule_card_ids)
            assert c.rules_status == "draft/unreviewed" and "Advisory only" in c.disclaimer
        drifts = [d.drift for c in report.countries for d in c.translation]
        assert drifts.count("increased") == 1
        assert not next(c for c in report.countries if c.country == "AU").translation

        # per-stage timings in the job JSON
        t = job["timings"]
        assert {"upload_probe", "stage_a", "translation", "total"} <= set(t)
        assert {"assessor:PK", "assessor:SG", "translation:PK"} <= set(t)
        assert all(v >= 0 for v in t.values()) and t["total"] >= t["stage_a"]
        assert "calls" in job  # mock makes no model calls


def test_analyze_validation(tiny_video):
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0]).json()["video_id"]
        assert client.post("/api/analyze", json={"video_id": video_id, "countries": []}).status_code == 422
        assert client.post("/api/analyze", json={"video_id": video_id, "countries": ["US"]}).status_code == 400
        assert client.post("/api/analyze", json={"video_id": "f" * 32, "countries": ["IN"]}).status_code == 404
        assert client.get("/api/jobs/" + "a" * 32).status_code == 404
        assert client.get("/api/jobs/../../etc").status_code == 404


def test_job_error_is_reported(tiny_video, monkeypatch):
    from app import pipeline

    def boom(*a, **kw):
        raise RuntimeError("inventory exploded")

    monkeypatch.setattr(pipeline, "stage_a", boom)
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0]).json()["video_id"]
        job_id = client.post("/api/analyze", json={"video_id": video_id, "countries": ["IN"]}).json()["job_id"]
        job = _wait(client, job_id)
        assert job["status"] == "error" and "inventory exploded" in job["error"]
        assert next(s for s in job["stages"] if s["key"] == "inventory")["status"] == "error"


def test_access_code_enforced(tiny_video, monkeypatch):
    monkeypatch.setenv("SC_ACCESS_CODE", "letmein")
    with TestClient(app) as client:
        for path in ["/api/countries", "/api/status", "/api/jobs/" + "a" * 32,
                     "/api/videos/" + "a" * 32, "/api/laws/IN/pdf"]:
            assert client.get(path).status_code == 401, path
        assert client.get("/api/countries", headers={"X-Access-Code": "wrong"}).status_code == 401
        assert _upload(client, tiny_video[0]).status_code == 401
        assert client.get("/api/countries", headers={"X-Access-Code": "letmein"}).status_code == 200
        client.cookies.set("sc_access", "letmein")
        assert client.get("/api/countries").status_code == 200
        assert _upload(client, tiny_video[0]).status_code == 200
        # static UI stays reachable so the user can enter the code
        client.cookies.clear()
        assert client.get("/").status_code == 200


def test_static_ui_served():
    with TestClient(app) as client:
        html = client.get("/").text
        assert "StreamClearance" in html and "/app.js" in html
        js = client.get("/app.js").text
        # model text is only ever inserted as text, never parsed as HTML
        assert not re.search(r"(inner|outer)HTML\s*=|insertAdjacentHTML|document\.write", js)


def test_law_pdf_per_registry_law():
    client = TestClient(app)
    r = client.get("/api/laws/MY/MAS1956/pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert client.get("/api/laws/MY/NOPE/pdf").status_code == 404
    assert client.get("/api/laws/IN/MAS1956/pdf").status_code == 404


def test_countries_coverage_fields(real_provisions):
    with TestClient(app) as client:
        cs = {c["code"]: c for c in client.get("/api/countries").json()}
    my, sg, au, pk = cs["MY"], cs["SG"], cs["AU"], cs["PK"]
    assert my["coverage"] == au["coverage"] == sg["coverage"] == "deep" and pk["coverage"] == "basic"
    assert my["laws_count"] == len(my["laws"]) == 5
    assert my["provisions_count"] == sum(law["provisions"] for law in my["laws"]) > 0
    assert au["laws_count"] == 3 and sg["laws_count"] == 2
    assert au["provisions_count"] == sum(law["provisions"] for law in au["laws"]) > 0
    assert pk["provisions_count"] == 0 and pk["laws_count"] == 1
    assert sg["bill"] is False and my["bill"] is False
    assert all(c["draft"] == (c["rules_status"] != "reviewed") for c in cs.values())
    assert my["laws"][0]["pdf_url"].startswith("/api/laws/MY/")


def test_countries_basic_without_provisions():
    with TestClient(app) as client:
        cs = client.get("/api/countries").json()
    assert all(c["coverage"] == "basic" and c["provisions_count"] == 0 for c in cs)


def test_taxonomy_labels():
    with TestClient(app) as client:
        t = client.get("/api/taxonomy").json()
    assert t["leaves"]["gambling.promotion"] == "Gambling promotion"
    assert t["categories"]["illegal_goods_or_gambling"] == "Illegal goods or gambling"


def test_video_meta(tiny_video):
    with TestClient(app) as client:
        up = _upload(client, tiny_video[0]).json()
        meta = client.get(f"/api/videos/{up['video_id']}/meta").json()
        assert meta == up and meta["filename"] == "clip.mp4"
        assert client.get("/api/videos/" + "a" * 32 + "/meta").status_code == 404


def _patched_inventory(monkeypatch, *, speech: bool, leaves: bool):
    from app import pipeline

    real = pipeline.mock_inventory

    def fake(duration_s):
        inv = real(duration_s)
        for s in inv.segments:
            s.transcript = s.transcript if speech else ""
            s.leaves = s.leaves if leaves else []
            s.categories = s.categories if leaves else []
            s.idioms = s.idioms if speech else []
            s.claims = s.claims if leaves else []
        return inv

    monkeypatch.setattr(pipeline, "mock_inventory", fake)


def test_clean_video_with_speech_gets_drift_for_every_drift_country(tiny_video, monkeypatch):
    _patched_inventory(monkeypatch, speech=True, leaves=False)
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0]).json()["video_id"]
        job_id = client.post("/api/analyze", json={"video_id": video_id,
                                                   "countries": ["MY", "IN", "AU"]}).json()["job_id"]
        job = _wait(client, job_id)
        assert job["status"] == "done", job
        by = {c["country"]: c for c in job["report"]["countries"]}
        assert all(not c["findings"] for c in by.values())
        assert by["MY"]["translation"] and by["IN"]["translation"] and not by["AU"]["translation"]
        tr = next(s for s in job["stages"] if s["key"] == "translation")
        assert tr["status"] == "done"
        riskier = [c for c in by.values() if any(d["drift"] == "increased" for d in c["translation"])]
        assert len(riskier) == 1 and riskier[0]["badge"] == "amber"


def test_no_speech_video_says_so(tiny_video, monkeypatch):
    _patched_inventory(monkeypatch, speech=False, leaves=True)
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0]).json()["video_id"]
        job_id = client.post("/api/analyze", json={"video_id": video_id,
                                                   "countries": ["MY", "AU"]}).json()["job_id"]
        job = _wait(client, job_id)
        assert job["status"] == "done", job
        tr = next(s for s in job["stages"] if s["key"] == "translation")
        assert (tr["status"], tr["detail"]) == ("skipped", "no speech to translate")
        by = {c["country"]: c for c in job["report"]["countries"]}
        assert by["MY"]["translation"] == [] and by["MY"]["translation_note"] == "No speech to translate."


def test_drift_starts_before_the_countries_are_assessed(tiny_video, monkeypatch):
    from app import pipeline

    order = []
    real_drift, real_b = pipeline.run_drift, pipeline.stage_b

    async def drift(*a, **kw):
        order.append("drift")
        return await real_drift(*a, **kw)

    def stage_b(*a, **kw):
        order.append("country")
        return real_b(*a, **kw)

    monkeypatch.setattr(pipeline, "run_drift", drift)
    monkeypatch.setattr(pipeline, "stage_b", stage_b)
    with TestClient(app) as client:
        video_id = _upload(client, tiny_video[0]).json()["video_id"]
        job = _wait(client, client.post("/api/analyze", json={
            "video_id": video_id, "countries": ["MY", "SG"]}).json()["job_id"])
    assert job["status"] == "done"
    assert order[0] == "drift" and order.count("country") == 2
    assert all(c["translation"] for c in job["report"]["countries"])


def test_country_without_verified_law_is_not_checked(tiny_video, real_provisions):
    with TestClient(app) as client:
        cs = {c["code"]: c for c in client.get("/api/countries").json()}
        assert cs["PK"]["checked"] is False
        assert all(cs[cc]["checked"] for cc in ("IN", "AU", "MY", "SG"))
        video_id = _upload(client, tiny_video[0]).json()["video_id"]
        job = _wait(client, client.post("/api/analyze", json={
            "video_id": video_id, "countries": ["SG", "PK"]}).json()["job_id"])
        assert job["status"] == "done", job
        by = {c["country"]: c for c in job["report"]["countries"]}
        assert by["PK"]["overall_risk"] == "not_checked" and by["PK"]["badge"] == "not_checked"
        assert by["SG"]["overall_risk"] != "not_checked"
        pk_stage = next(s for s in job["stages"] if s["key"] == "country:PK")
        assert pk_stage["detail"].startswith("not_checked")
