"""FastAPI app: upload, analysis jobs, country/law metadata and the static UI."""
from __future__ import annotations

import hmac
import logging
import uuid
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, jobs, media
from app.storage import TooLarge, get_storage

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="StreamClearance")
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
            "max_duration_s": config.MAX_DURATION_S}


def _rules_status(c: config.CountryLaw) -> str:
    if c.rules_path.exists():
        from app.pipeline import load_cards
        return load_cards(c.code)[1]
    return "draft/unreviewed"


@app.get("/api/countries")
def countries():
    return [
        {
            "code": c.code,
            "name": c.name,
            "law": c.law,
            "label": c.label,
            "drift_languages": list(c.drift_languages),
            "rules_status": _rules_status(c),
            "pdf_url": f"/api/laws/{c.code}/pdf",
        }
        for c in config.COUNTRIES.values()
    ]


@app.get("/api/laws/{code}/pdf")
def law_pdf(code: str):
    c = config.COUNTRIES.get(code.upper())
    if c is None or not c.pdf_path.exists():
        raise HTTPException(404, "law not found")
    return FileResponse(c.pdf_path, media_type="application/pdf",
                        filename=c.pdf_path.name, content_disposition_type="inline")


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


class AnalyzeRequest(BaseModel):
    video_id: str
    countries: list[str] = Field(min_length=1)


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    store = get_storage()
    if store.load_video_meta(req.video_id) is None or store.video_path(req.video_id) is None:
        raise HTTPException(404, "video not found")
    codes = list(dict.fromkeys(c.upper() for c in req.countries))
    unknown = [c for c in codes if c not in config.COUNTRIES]
    if unknown:
        raise HTTPException(400, f"unknown countries: {', '.join(unknown)}")
    job = jobs.create_job(req.video_id, codes)
    jobs.start_job(job)
    return {"job_id": job["job_id"]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = get_storage().load_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
