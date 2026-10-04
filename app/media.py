"""Video metadata via ffprobe (optional: duration is None when ffprobe is missing)."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from app import config

log = logging.getLogger(__name__)


class UnreadableVideo(ValueError):
    pass


def probe_duration(path: Path) -> float | None:
    """Return the duration in seconds, None if ffprobe is unavailable.

    Raises UnreadableVideo when ffprobe exists but can't read a duration, which
    means the upload isn't a real video despite its extension.
    """
    ffprobe = config.ffprobe_path()
    if not ffprobe:
        log.warning("ffprobe not found; skipping duration check for %s", path.name)
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        return float(json.loads(out)["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, ValueError, TypeError) as e:
        raise UnreadableVideo("could not read video duration") from e
