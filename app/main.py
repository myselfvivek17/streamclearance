"""FastAPI app: upload, analysis jobs, country/law metadata and the static UI."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app import config, media
from app.storage import TooLarge, get_storage

log = logging.getLogger(__name__)

app = FastAPI(title="StreamClearance")

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
