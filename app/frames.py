"""Moment thumbnails: one small JPEG still per flagged moment, taken before the video is deleted.

After stage A, `extract` takes one frame at the midpoint of every segment stage A
tagged with a taxonomy leaf (only those can become findings), ~320 px wide, and
stores it with the job (`jobs/<id>/frames/<start ms>.jpg`). The frames are derived
from the video, so they are report data: served behind the access gate and
deleted with the report (app/storage.py `delete_job`). ffmpeg is optional: when
it is missing or fails, there is simply no thumbnail.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from app import config
from app.schemas import Inventory
from app.storage import get_storage

log = logging.getLogger(__name__)

WIDTH = 320
QUALITY = 5  # ffmpeg -q:v, 2 (best) .. 31
NAME = re.compile(r"^\d{1,9}\.jpg$")
TIMEOUT_S = 20


def frame_name(start_s: float) -> str:
    return f"{int(round(start_s * 1000))}.jpg"


def valid_name(name: str) -> bool:
    return bool(NAME.match(name))


def extract(video_path: Path, job_id: str, inventory: Inventory) -> dict[float, str]:
    """Frames for the flagged segments -> {segment start_s: file name}. Never raises."""
    ffmpeg = config.ffmpeg_path()
    flagged = [s for s in inventory.segments if s.leaves]
    if not ffmpeg or not flagged:
        return {}
    out_dir = get_storage().frames_dir(job_id)
    if out_dir is None:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    made: dict[float, str] = {}
    for s in flagged:
        name = frame_name(s.start_s)
        mid = (s.start_s + s.end_s) / 2
        try:
            subprocess.run(
                [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{mid:.2f}", "-i", str(video_path),
                 "-frames:v", "1", "-vf", f"scale={WIDTH}:-2", "-q:v", str(QUALITY), "-y",
                 str(out_dir / name)],
                capture_output=True, timeout=TIMEOUT_S, check=True)
        except (OSError, subprocess.SubprocessError) as e:
            log.warning("job %s: frame at %.1fs failed (%s)", job_id, mid, type(e).__name__)
            continue
        if (out_dir / name).exists() and (out_dir / name).stat().st_size > 0:
            made[s.start_s] = name
    if get_storage().gone_reason(job_id):  # the report was deleted while ffmpeg ran
        shutil.rmtree(out_dir.parent, ignore_errors=True)
        return {}
    log.info("job %s: %d of %d frames taken", job_id, len(made), len(flagged))
    return made


def attach(inventory: Inventory, made: dict[float, str]) -> None:
    for s in inventory.segments:
        if s.start_s in made:
            s.frame = made[s.start_s]
