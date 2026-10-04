"""Analysis pipeline.

Stage A: video -> Inventory (one Gemini video call per video).

Every stage has a deterministic mock used when `config.mock_mode()` is true
(no GEMINI_API_KEY, or SC_MOCK=1), so the whole app runs without a key.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

from pydantic import BaseModel

from app import config, gemini
from app.schemas import Category, Inventory, Segment

log = logging.getLogger(__name__)

MOCK_UNKNOWN_DURATION_S = 60.0


# ================================================================ stage A

class StageADraft(BaseModel):
    """What Gemini returns for stage A; duration comes from ffprobe, not the model."""
    languages: list[str] = []
    segments: list[Segment] = []


STAGE_A_SYSTEM = f"""You build a factual, timestamped inventory of a short video for a \
content-compliance tool. Do not judge legality; only describe what is there.

Split the video into consecutive segments of roughly 3-15 seconds at natural breaks \
(sentence or scene changes). For each segment give:
- start_s / end_s in seconds from the start of the video (no gaps, no overlaps)
- transcript: verbatim speech in the original language ("" if none)
- language: ISO 639-1 code of the speech ("" if none)
- claims: factual or health/financial claims made, one per item
- visuals: what is shown (people, actions, symbols, products, logos, flags, weapons, etc.)
- on_screen_text: captions, overlays or signs, verbatim
- categories: zero or more of the fixed list below that the segment might plausibly touch. \
Leave empty for clearly neutral segments. Use "other" only when nothing else fits.

Categories: {", ".join(c.value for c in Category)}"""


def stage_a(video_path: Path, duration_s: float | None = None, *, mock: bool | None = None) -> Inventory:
    """Video -> Inventory. duration_s comes from ffprobe and may be unknown (None)."""
    mock = config.mock_mode() if mock is None else mock
    if mock:
        return mock_inventory(duration_s)

    mime = config.VIDEO_TYPES.get(video_path.suffix.lower(), "video/mp4")
    dur_hint = f"The video is {duration_s:.1f} seconds long. " if duration_s else ""
    raw = gemini.generate_json_with_video(
        video_path, mime,
        f"{dur_hint}Return the inventory as JSON.",
        StageADraft,
        model=config.SC_MODEL_FAST,
        system_instruction=STAGE_A_SYSTEM,
    )
    draft = StageADraft.model_validate(raw)
    segments = sorted(draft.segments, key=lambda s: s.start_s)
    if duration_s is None:
        duration_s = max((s.end_s for s in segments), default=0.0)
    kept = []
    for s in segments:
        if s.start_s >= duration_s:
            log.warning("stage A: dropping segment at %.1fs past duration %.1fs", s.start_s, duration_s)
            continue
        kept.append(s.model_copy(update={"end_s": min(s.end_s, duration_s)}))
    return Inventory(duration_s=duration_s, languages=draft.languages, segments=kept)


_MOCK_SEGMENTS = [
    dict(transcript="Hey everyone, welcome back! Today I'm unboxing the new festival snack box.",
         visuals="Creator at a kitchen table holding a colourful snack box.",
         on_screen_text="FESTIVAL SNACK BOX UNBOXING", claims=[], categories=[]),
    dict(transcript="Honestly, fasting this month is pointless. Just eat the snacks, God won't mind.",
         visuals="Creator laughing, waving a pastry in front of a religious festival poster.",
         on_screen_text="", claims=["Religious fasting is pointless"],
         categories=[Category.religious_offence]),
    dict(transcript="And these snacks cure diabetes. Doctors don't want you to know this.",
         visuals="Close-up of a snack label; creator points at it.",
         on_screen_text="CURES DIABETES?!", claims=["The snacks cure diabetes"],
         categories=[Category.misinformation_public_order]),
    dict(transcript="Use my code at LuckySpin Casino for a hundred free bets tonight.",
         visuals="Phone screen showing an online casino slot game.",
         on_screen_text="LUCKYSPIN.BET  CODE: SNACK100", claims=["100 free bets with code"],
         categories=[Category.illegal_goods_or_gambling]),
    dict(transcript="This deal is a killer, it's basically a crime not to grab it.",
         visuals="Creator holding the box up to the camera, smiling.",
         on_screen_text="LINK IN BIO", claims=[], categories=[]),
    dict(transcript="Like and subscribe, see you next week!",
         visuals="End card with channel logo.", on_screen_text="SUBSCRIBE", claims=[], categories=[]),
]


def mock_inventory(duration_s: float | None) -> Inventory:
    """Deterministic plausible inventory scaled to the real duration."""
    duration = duration_s if duration_s and duration_s > 0 else MOCK_UNKNOWN_DURATION_S
    n = max(3, min(8, math.ceil(duration / 10)))
    step = duration / n
    segments = []
    for i in range(n):
        tpl = _MOCK_SEGMENTS[i % len(_MOCK_SEGMENTS)]
        segments.append(Segment(
            start_s=round(i * step, 2),
            end_s=round(duration if i == n - 1 else (i + 1) * step, 2),
            language="en",
            **tpl,
        ))
    return Inventory(duration_s=round(duration, 2), languages=["en"], segments=segments)

