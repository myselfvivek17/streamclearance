"""Local file storage for uploads and job status.

Everything that touches `uploads/` or `jobs/` goes through `Storage`, so the
local implementation can be swapped for a GCS-backed one later without
changing the API or pipeline.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from app import config

_ID = re.compile(r"^[a-f0-9]{32}$")


class TooLarge(ValueError):
    pass


def valid_id(value: str) -> bool:
    """IDs are uuid4 hex; anything else is rejected before touching the disk."""
    return bool(_ID.match(value))


class LocalStorage:
    def __init__(self, root: Path):
        self.uploads = root / "uploads"
        self.jobs = root / "jobs"

    # ------------------------------------------------------------ videos

    async def save_video(
        self, video_id: str, ext: str, chunks: AsyncIterator[bytes], max_bytes: int
    ) -> Path:
        """Stream an upload to disk, aborting (and cleaning up) past max_bytes."""
        self.uploads.mkdir(parents=True, exist_ok=True)
        path = self.uploads / f"{video_id}{ext}"
        size = 0
        try:
            with path.open("wb") as fh:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > max_bytes:
                        raise TooLarge(f"file exceeds {max_bytes // (1024 * 1024)} MB")
                    fh.write(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    def video_path(self, video_id: str) -> Path | None:
        if not valid_id(video_id):
            return None
        for ext in config.VIDEO_TYPES:
            p = self.uploads / f"{video_id}{ext}"
            if p.exists():
                return p
        return None

    def delete_video(self, video_id: str) -> int:
        """Delete the video, its meta and anything else stored under its id
        (uploads/<id>*). Returns the number of files removed."""
        if not valid_id(video_id) or not self.uploads.exists():
            return 0
        n = 0
        for p in self.uploads.glob(f"{video_id}*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            n += 1
        return n

    def video_ids(self) -> dict[str, float]:
        """Every stored video id (video or meta present) -> oldest file mtime."""
        out: dict[str, float] = {}
        if not self.uploads.exists():
            return out
        for p in self.uploads.iterdir():
            vid = p.name[:32]
            if valid_id(vid):
                mtime = p.stat().st_mtime
                out[vid] = min(out.get(vid, mtime), mtime)
        return out

    def save_video_meta(self, video_id: str, meta: dict[str, Any]) -> None:
        self._write_json(self.uploads / f"{video_id}.meta.json", meta)

    def load_video_meta(self, video_id: str) -> dict[str, Any] | None:
        if not valid_id(video_id):
            return None
        return self._read_json(self.uploads / f"{video_id}.meta.json")

    # ------------------------------------------------------------ jobs

    def save_job(self, job_id: str, data: dict[str, Any]) -> None:
        self._write_json(self.jobs / f"{job_id}.json", data)

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        if not valid_id(job_id):
            return None
        return self._read_json(self.jobs / f"{job_id}.json")

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        """Atomic write so a polling reader never sees a half-written file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


_storage: LocalStorage | None = None


def get_storage() -> LocalStorage:
    global _storage
    if _storage is None:
        _storage = LocalStorage(config.STORAGE_DIR)
    return _storage
