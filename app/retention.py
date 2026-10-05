"""Data retention: what is deleted, and when.

- The uploaded video (and everything stored under its id: meta, derived files,
  and any upload-bucket object made for the Gemini call) is deleted as soon as
  its analysis job ends, whether it finished, failed or was cancelled.
- A video that is uploaded but never analysed is deleted after
  SC_UNUSED_VIDEO_TTL_MIN minutes.
- A report (the job record) is deleted SC_REPORT_TTL_DAYS days after the job
  was created, by the sweeper and lazily when it is read, or at once on
  DELETE /api/jobs/{id}. A content-free marker (id + reason) is kept for as
  long again, so the UI can say "expired" rather than "not found".
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from app import config
from app.storage import get_storage

log = logging.getLogger(__name__)


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _may_have_gcs_copy(size: int | None) -> bool:
    """Only Vertex calls on videos too large to inline go through the upload bucket."""
    return (config.provider() == "gemini" and config.gemini_backend() == "vertex"
            and (size is None or size > config.VERTEX_INLINE_MAX_BYTES))


def purge_video(video_id: str) -> str:
    """Delete everything stored for this video, locally and in GCS. Returns the
    deletion time (ISO 8601 UTC). Never raises: failures are logged (ids only)."""
    store = get_storage()
    path = store.video_path(video_id)
    size = path.stat().st_size if path is not None and path.exists() else None
    meta = store.load_video_meta(video_id) or {}
    size = size if size is not None else meta.get("size_bytes")
    n_local = store.delete_video(video_id)
    n_gcs = 0
    if _may_have_gcs_copy(size):
        from app import gemini

        try:
            n_gcs = gemini.delete_gcs_objects(video_id)
        except Exception as e:  # noqa: BLE001 - the bucket's lifecycle rule is the backstop
            log.warning("video %s: GCS cleanup failed (%s)", video_id, type(e).__name__)
    log.info("video %s deleted: %d local files, %d GCS objects", video_id, n_local, n_gcs)
    return iso(time.time())


def sweep_unused_videos(active: set[str], now: float | None = None) -> int:
    """Delete videos older than SC_UNUSED_VIDEO_TTL_MIN that no running job uses."""
    now = time.time() if now is None else now
    cutoff = now - config.unused_video_ttl_s()
    n = 0
    for video_id, mtime in get_storage().video_ids().items():
        if video_id not in active and mtime < cutoff:
            purge_video(video_id)
            n += 1
    return n


def report_expired(job: dict, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    return now >= job["created_at"] + config.report_ttl_days() * 86400


def delete_report(job_id: str, reason: str) -> None:
    """Delete a job record, its video (if still there) and every stored file for it."""
    store = get_storage()
    job = store.load_job(job_id)
    if job is not None:
        purge_video(job["video_id"])
    store.delete_job(job_id, reason)
    log.info("job %s deleted (%s)", job_id, reason)


def expire_reports(active: set[str], now: float | None = None) -> int:
    """Delete reports past SC_REPORT_TTL_DAYS (skipping running jobs)."""
    now = time.time() if now is None else now
    store = get_storage()
    n = 0
    for job_id in store.job_ids():
        try:
            job = store.load_job(job_id)
        except ValueError:  # unreadable record: leave it for a person to look at
            log.warning("job %s: unreadable record", job_id)
            continue
        if job is not None and job_id not in active and report_expired(job, now):
            delete_report(job_id, "expired")
            n += 1
    store.sweep_gone(now - config.report_ttl_days() * 86400)
    return n
