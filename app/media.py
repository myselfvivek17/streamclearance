"""Video metadata via ffprobe (optional: duration is None when ffprobe is missing)."""
from __future__ import annotations

import json
import logging
import re
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


# Tag values naming an AI video/image/voice tool. Only these keys are read: they are where
# encoders and editing tools write their name (format and stream tags).
AI_TOOL = re.compile(
    r"\b(sora|veo|runway|pika|kling|luma|dream ?machine|synthesia|heygen|d-id|elevenlabs|"
    r"midjourney|stable ?diffusion|dall-?e|imagen|firefly|hailuo|minimax|ai[- ]generated|"
    r"made with ai|generated (?:by|with) ai)\b", re.IGNORECASE)
AI_TAG_KEYS = ("comment", "encoder", "handler_name", "software", "description", "title")


def ai_metadata_hints(path: Path) -> list[str]:
    """Metadata tags that name an AI tool, e.g. 'encoder tag mentions "Veo"'. [] when
    ffprobe is missing or the tags say nothing. Only the matched tool name is kept."""
    ffprobe = config.ffprobe_path()
    if not ffprobe:
        return []
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True).stdout
        data = json.loads(out)
    except (subprocess.SubprocessError, OSError, ValueError):
        return []
    tags = [data.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in data.get("streams", [])]
    hints: list[str] = []
    for t in tags:
        for key, value in (t or {}).items():
            if key.lower() not in AI_TAG_KEYS:
                continue
            m = AI_TOOL.search(str(value))
            hint = f"file metadata: {key.lower()} tag mentions \"{m.group(0)}\"" if m else None
            if hint and hint not in hints:
                hints.append(hint)
    return hints
