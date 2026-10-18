"""FastAPI app: upload, analysis jobs, country/law metadata and the static UI."""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import html
import logging
import time
import uuid
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, frames, jobs, logsafe, media, provisions, retention, taxonomy
from app.schemas import Category
from app.storage import TooLarge, get_storage

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logsafe.install()  # no content in logs, including library logs and tracebacks
config.apply_adk_env()  # ADK agents use the same Gemini backend as app/gemini.py

SWEEP_EVERY_S = 30 * 60


def sweep() -> None:
    """Delete videos nobody analysed within SC_UNUSED_VIDEO_TTL_MIN and reports
    older than SC_REPORT_TTL_DAYS."""
    videos = retention.sweep_unused_videos(jobs.active_video_ids())
    reports = retention.expire_reports(jobs.active_job_ids())
    log.info("sweep: %d unused videos, %d expired reports deleted", videos, reports)


async def _sweeper() -> None:
    while True:
        try:
            await asyncio.to_thread(sweep)
        except Exception as e:  # noqa: BLE001 - keep sweeping
            log.warning("sweep failed (%s)", type(e).__name__)
        await asyncio.sleep(SWEEP_EVERY_S)


def warm_up() -> None:
    """Load what the first job would otherwise pay for (Cloud Run scales to zero):
    provisions and the lookup index, taxonomy, the ADK and genai libraries, the
    Gemini client and, on Vertex, the auth session used by law search. No model calls."""
    t0 = time.perf_counter()
    provisions.all_provisions()
    from app import retrieval

    retrieval.lookup(next(iter(config.COUNTRIES)), [])
    taxonomy.prompt_list()
    import google.adk.runners  # noqa: F401 - import cost (~1s) paid here
    import google.adk.workflow  # noqa: F401
    if not config.mock_mode():
        from app import gemini

        gemini._get_client()
        if config.law_search() == "rag":
            from app import law_search

            law_search._get_session()
    log.info("warm-up done in %.1fs", time.perf_counter() - t0)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await asyncio.to_thread(warm_up)
    except Exception as e:  # noqa: BLE001 - the first job will load it instead
        log.warning("warm-up failed (%s)", type(e).__name__)
    task = asyncio.create_task(_sweeper())  # runs once on startup, then every 30 min
    yield
    task.cancel()


app = FastAPI(title="StreamClearance", lifespan=lifespan)
STATIC_DIR = Path(__file__).parent / "static"
ACCESS_HEADER = "X-Access-Code"
ACCESS_COOKIE = "sc_access"


@app.middleware("http")
async def access_code_gate(request: Request, call_next):
    """If SC_ACCESS_CODE is set, every /api route needs it (header or cookie)."""
    code = config.access_code()
    if code and request.url.path.startswith("/api/"):
        given = request.headers.get(ACCESS_HEADER) or unquote(request.cookies.get(ACCESS_COOKIE, ""))
        if not hmac.compare_digest(given.encode(), code.encode()):
            return JSONResponse({"detail": "access code required"}, status_code=401)
    return await call_next(request)


@app.get("/api/status")
def status():
    return {"mock": config.mock_mode(), "max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_duration_s": config.MAX_DURATION_S,
            # What the upload page may promise (each one true in code: app/retention.py, config.no_training)
            "privacy": {"video_deleted_after_check": True, "report_ttl_days": config.report_ttl_days(),
                        "no_training": config.no_training()}}


def _duration(value: float, unit: str) -> str:
    n = f"{value:g}"
    return f"{n} {unit}{'' if n == '1' else 's'}"


def privacy_values() -> dict[str, str]:
    """The configuration-dependent sentences of the /privacy page."""
    region = config.app_region()
    backend = None if config.mock_mode() else config.gemini_backend()
    if backend == "vertex":
        loc = config.vertex_location()
        gemini_where = f"Google’s Gemini model on Vertex AI, in Google Cloud location <b>{html.escape(loc)}</b>"
        if loc == "global":
            gemini_where += (": Google may process it in any of its regions, so there is no data residency"
                             " guarantee for this step.")
        else:
            gemini_where += " (Singapore)." if loc == "asia-southeast1" else "."
    elif backend == "studio":
        gemini_where = "Google’s Gemini model through the Gemini API (Google AI Studio)."
    else:
        gemini_where = "none. This copy runs in demo mode and sends nothing to a model."
    return {
        "unused_ttl": _duration(config.unused_video_ttl_s() / 60, "minute"),
        "report_ttl": _duration(config.report_ttl_days(), "day"),
        "app_where": (f"Google Cloud Run in Singapore (<b>{html.escape(region)}</b>)" if region == "asia-southeast1"
                      else f"Google Cloud Run, region <b>{html.escape(region)}</b>" if region
                      else "the server this copy of the app runs on"),
        "gemini_where": gemini_where,
        "training": ("Your content is never used to train AI models. Google’s terms for Vertex AI forbid"
                     " training on customer data without permission." if config.no_training() else
                     "This instance uses the Gemini API, whose terms depend on the plan: check them before"
                     " uploading anything confidential."),
        "frames": ("<b>Still frames</b>: before your video is deleted, one small still image is taken"
                   " from each flagged moment and shown next to the findings. The stills are part of the"
                   f" report and are deleted with it, after {_duration(config.report_ttl_days(), 'day')}"
                   " or when you delete the report." if config.ffmpeg_path() else
                   "<b>Still frames</b>: none are kept. This copy of the app keeps no images from your video."),
        "caching": ("Caching of requests by Vertex AI is turned off for this project." if config.vertex_cache_disabled()
                    else "Vertex AI may keep requests in memory (not on disk) for up to 24 hours to answer"
                         " repeated requests faster, separated from other customers."),
    }


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page():
    page = (STATIC_DIR.parent / "templates" / "privacy.html").read_text(encoding="utf-8")
    for key, value in privacy_values().items():
        page = page.replace("{{" + key + "}}", value)
    return page


def _rules_status(c: config.CountryLaw) -> str:
    if provisions.has_provisions(c.code) or c.rules_path.exists():
        from app.pipeline import load_cards
        return load_cards(c.code)[1]
    return "draft/unreviewed"


def _country(c: config.CountryLaw) -> dict:
    laws = provisions.law_infos(c.code)
    deep = provisions.has_provisions(c.code)  # basic = legacy rule cards only
    status = _rules_status(c)
    return {
        "code": c.code,
        "name": c.name,
        "law": c.law,
        "label": c.label,
        "drift_languages": list(c.drift_languages),
        "rules_status": status,
        "pdf_url": f"/api/laws/{c.code}/pdf",
        "coverage": "deep" if deep else "basic",
        "checked": provisions.checkable(c.code),  # False: no verified law loaded, nothing is checked
        "laws_count": len(laws),
        "provisions_count": sum(law.provisions for law in laws),
        "draft": status != "reviewed",
        "bill": any(law.status == "bill" for law in laws),
        "laws": [law.model_dump() for law in laws],
    }


@app.get("/api/countries")
def countries():
    return [_country(c) for c in config.COUNTRIES.values()]


CATEGORY_LABELS = {Category.synthetic_unlabelled: "AI-generated content without a label"}


@app.get("/api/taxonomy")
def get_taxonomy():
    """Labels for taxonomy leaves (report `leaf`) and legacy categories (report `category`)."""
    leaves = taxonomy.leaves().values()
    return {
        "leaves": {leaf.id: leaf.label for leaf in leaves},
        "categories": {cat.value: CATEGORY_LABELS.get(cat, cat.value.replace("_or_", " or ")
                                                      .replace("_", " ").capitalize())
                       for cat in Category},
    }


@app.get("/api/laws/{code}/pdf")
def law_pdf(code: str):
    c = config.COUNTRIES.get(code.upper())
    if c is None or not c.pdf_path.exists():
        raise HTTPException(404, "law not found")
    return FileResponse(c.pdf_path, media_type="application/pdf",
                        filename=c.pdf_path.name, content_disposition_type="inline")


@app.get("/api/laws/{code}/{law_id}/pdf")
def provision_law_pdf(code: str, law_id: str):
    """One law of a country's registry (a country can have several)."""
    law = next((x for x in provisions.laws_for(code.upper()) if x.id == law_id), None)
    if law is None or not law.pdf_path.exists():
        raise HTTPException(404, "law not found")
    return FileResponse(law.pdf_path, media_type="application/pdf",
                        filename=law.pdf_path.name, content_disposition_type="inline")


UPLOAD_CHUNK = 1024 * 1024


async def _chunks(file: UploadFile):
    while chunk := await file.read(UPLOAD_CHUNK):
        yield chunk


@app.post("/api/upload")
async def upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.VIDEO_TYPES:
        raise HTTPException(415, f"unsupported file type; use {', '.join(config.VIDEO_TYPES)}")
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype and not (ctype.startswith("video/") or ctype == "application/octet-stream"):
        raise HTTPException(415, f"unsupported content type {ctype!r}")

    store = get_storage()
    video_id = uuid.uuid4().hex
    try:
        path = await store.save_video(video_id, ext, _chunks(file), config.MAX_UPLOAD_BYTES)
    except TooLarge as e:
        raise HTTPException(413, str(e)) from e

    t0 = time.perf_counter()
    try:
        duration = media.probe_duration(path)
    except media.UnreadableVideo as e:
        store.delete_video(video_id)
        raise HTTPException(400, str(e)) from e
    if duration is not None and duration > config.MAX_DURATION_S:
        store.delete_video(video_id)
        raise HTTPException(
            400, f"video is {duration:.0f}s; the limit is {config.MAX_DURATION_S / 60:.0f} minutes")

    meta = {
        "video_id": video_id,
        "filename": Path(file.filename or "").name[:200],
        "ext": ext,
        "mime_type": config.VIDEO_TYPES[ext],
        "size_bytes": path.stat().st_size,
        "duration_s": duration,
        "probe_s": round(time.perf_counter() - t0, 2),
    }
    store.save_video_meta(video_id, meta)
    log.info("upload %s %s %d bytes duration=%s", video_id, ext, meta["size_bytes"], duration)
    return meta


@app.get("/api/videos/{video_id}")
def get_video(video_id: str):
    store = get_storage()
    path = store.video_path(video_id)
    meta = store.load_video_meta(video_id)
    if path is None or meta is None:
        raise HTTPException(404, "video not found")
    return FileResponse(path, media_type=meta["mime_type"])


@app.get("/api/videos/{video_id}/meta")
def get_video_meta(video_id: str):
    """Upload metadata (file name, size, duration), so the UI can restore state after a reload."""
    meta = get_storage().load_video_meta(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    return meta


class AnalyzeRequest(BaseModel):
    video_id: str
    countries: list[str] = Field(min_length=1)


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    store = get_storage()
    if store.load_video_meta(req.video_id) is None or store.video_path(req.video_id) is None:
        raise HTTPException(404, "video not found")
    if req.video_id in jobs.active_video_ids():
        raise HTTPException(409, "this video is already being analysed")
    codes = list(dict.fromkeys(c.upper() for c in req.countries))
    unknown = [c for c in codes if c not in config.COUNTRIES]
    if unknown:
        raise HTTPException(400, f"unknown countries: {', '.join(unknown)}")
    job = jobs.create_job(req.video_id, codes)
    jobs.start_job(job)
    return {"job_id": job["job_id"]}


EXPIRED = "This report expired and was deleted"
GONE = {"expired": EXPIRED, "deleted": "This report was deleted"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, since: int = 0):
    """The job and its report. `events` is the activity feed; with `?since=n` only the
    events after the first n (`events_total` says how many there are)."""
    store = get_storage()
    job = store.load_job(job_id)
    if job is not None and job_id not in jobs.active_job_ids() and retention.report_expired(job):
        retention.delete_report(job_id, "expired")  # lazily, between sweeps
        job = None
    if job is None:
        raise HTTPException(404, GONE.get(store.gone_reason(job_id) or "", "job not found"))
    feed = job.get("events") or []
    job["events_total"] = len(feed)
    job["events"] = feed[max(0, since):]
    return job


@app.get("/api/jobs/{job_id}/frames/{name}")
def get_frame(job_id: str, name: str):
    """A moment thumbnail of this report (behind the access gate, deleted with the report)."""
    store = get_storage()
    job = store.load_job(job_id)
    if job is None or retention.report_expired(job) or not frames.valid_name(name):
        raise HTTPException(404, "frame not found")
    folder = store.frames_dir(job_id)
    path = folder / name if folder else None
    if path is None or not path.is_file():
        raise HTTPException(404, "frame not found")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a report now, with its video and everything stored for it."""
    store = get_storage()
    if store.load_job(job_id) is None:
        raise HTTPException(404, GONE.get(store.gone_reason(job_id) or "", "job not found"))
    await jobs.delete_job(job_id)
    return {"deleted": job_id}


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    """Delete an uploaded video that has not been analysed (e.g. "Replace video")."""
    store = get_storage()
    if store.load_video_meta(video_id) is None and store.video_path(video_id) is None:
        raise HTTPException(404, "video not found")
    if video_id in jobs.active_video_ids():
        raise HTTPException(409, "this video is being analysed; delete the report instead")
    retention.purge_video(video_id)
    return {"deleted": video_id}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
