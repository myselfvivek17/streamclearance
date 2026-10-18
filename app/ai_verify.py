"""Verification of stage A's signs of AI generation against frames of the video.

Stage A sometimes reports signs that are not there (a real AI clip: "garbled text on the
bottle label" when the bottle has no label, "six fingers" on a five-fingered hand). So
before any sign is shown or used, each visual sign is checked against still frames, in
one batched call, with a strict yes/no ("answer no when unsure"):

| sign | frames checked | if not confirmed |
|---|---|---|
| visual, with a time ("0:05 ...", "0:04-0:06 ...") | at the time, and 0.5 s either side (a range: start, middle, end) | dropped |
| visual, no time (e.g. "KlingAI watermark in the corner") | 4 spread over the video, plus the middle of up to 2 moments where stage A saw a visual sign | dropped |
| audio only (voice, narration, speech; no visual word) | none: a still frame cannot show a voice | kept |
| file metadata (an AI tool named in the file's tags, read by ffprobe) | none: read from the file, not by a model | kept |

Motion-only signs (morphing, flicker, lip sync) are kept only if the frames show them.
When the frames cannot be taken or the call fails, visual signs are dropped too: the
report shows nothing unverified. The likelihood is then recomputed from the signs left
(app/schemas.py SyntheticLikelihood scale): never raised, and at most "none" for no sign,
"low" for one, "medium" for two; file metadata still means at least "medium". Segment
level signs (stage A's per-moment list) are unverified, so only their audio signs are
kept. Unconfirmed signs are counted (`SyntheticCheck.unconfirmed`), never shown.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app import config, events, gemini
from app.schemas import Inventory, SyntheticCheck

log = logging.getLogger(__name__)

WIDTH = 960  # wide enough to count fingers or read a label
QUALITY = 4
TIMEOUT_S = 20
SPREAD_S = 0.5  # frames either side of a named time
MAX_UNTIMED_FRAMES = 4
LIKELIHOOD = ["none", "low", "medium", "high"]

_TIME = re.compile(r"(?<![\d:.])(\d{1,2}):([0-5]\d)(?:\.(\d))?(?:\s*(?:-|–|to)\s*(\d{1,2}):([0-5]\d)(?:\.(\d))?)?")
_TTS = re.compile(r"\btext[- ]to[- ]speech\b", re.IGNORECASE)
_AUDIO = re.compile(r"\b(voice|narrat|tts|speech|audio|prosody|intonation|inflection|breath|vocal|"
                    r"spoken|says|said|states|sound|gibberish|clon)", re.IGNORECASE)
_VISUAL = re.compile(r"\b(face|hand|finger|lip|mouth|teeth|blink|eye|skin|hair|frame|lighting|shadow|"
                     r"morph|artefact|artifact|image|footage|watermark|background|text|label|sign\b|"
                     r"signs\b|logo|screen|caption|overlay|scene|split|blur|warp|distort|object|body|"
                     r"limb|visual|shown|display|appear|look|letter|bottle)", re.IGNORECASE)


def is_audio(signal: str) -> bool:
    """A sign only about what is heard: a still frame cannot confirm or refute it."""
    text = _TTS.sub("tts", signal)
    return bool(_AUDIO.search(text)) and not _VISUAL.search(text)


def times(signal: str, duration_s: float) -> list[float]:
    """Frame times for a sign that names one (a time or a range), else []."""
    m = _TIME.search(signal)
    if not m:
        return []
    start = int(m[1]) * 60 + int(m[2]) + int(m[3] or 0) / 10
    if m[4] is not None:
        end = int(m[4]) * 60 + int(m[5]) + int(m[6] or 0) / 10
        ts = [start, (start + end) / 2, end] if end > start else [start]
    else:
        ts = [start - SPREAD_S, start, start + SPREAD_S]
    last = max(0.0, duration_s - 0.05)
    out: list[float] = []
    for t in ts:
        t = round(min(max(t, 0.0), last), 2)
        if t not in out:
            out.append(t)
    return out


def untimed_frames(inventory: Inventory) -> list[float]:
    """Frames for a sign that names no time: spread over the whole video, plus the middle
    of the first moments where stage A saw a visual sign."""
    dur = inventory.duration_s
    spread = [round((i + 0.5) * dur / MAX_UNTIMED_FRAMES, 2) for i in range(MAX_UNTIMED_FRAMES)]
    seen = [round((s.start_s + s.end_s) / 2, 2) for s in inventory.segments
            if any(not is_audio(x) for x in s.synthetic_signals)][:2]
    return sorted(set(spread + seen))


def frame_at(video_path: Path, t: float) -> bytes | None:
    ffmpeg = config.ffmpeg_path()
    if not ffmpeg:
        return None
    try:
        out = subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{t:.2f}", "-i", str(video_path),
             "-frames:v", "1", "-vf", f"scale='min({WIDTH},iw)':-2", "-q:v", str(QUALITY),
             "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, timeout=TIMEOUT_S, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("AI-sign frame at %.1fs failed (%s)", t, type(e).__name__)
        return None
    return out or None


class Verdict(BaseModel):
    index: int
    observed: str = ""
    visible: bool = False


SYSTEM = """You check claimed signs that a video is AI-generated against still frames \
taken from it. Each claim lists the frames to look at (taken at the time it names, or \
across the video when it names none).

For each claim, first write `observed`: one short sentence on what those frames actually \
show for it (e.g. count the fingers; say whether the object has a label at all). Then \
`visible`: true ONLY if the claimed detail is clearly visible, as described, in at least \
one of its frames. Answer false when you are unsure, when the frames show something \
different (five fingers is not six; a bottle with no label cannot have garbled label \
text), when the detail is too small or blurred to tell, or when it is a motion sign \
(morphing, flicker, lip sync) that these frames do not show. One item per claim."""


def _clock(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:04.1f}"


def ask(video_path: Path, claims: list[tuple[str, list[float]]]) -> dict[int, Verdict]:
    """One call: every claim with its frames -> {claim index: verdict}. Claims whose
    frames could not be taken get no verdict (not confirmed)."""
    wanted = sorted({t for _, ts in claims for t in ts})
    with ThreadPoolExecutor(max_workers=4) as pool:
        got = dict(zip(wanted, pool.map(lambda t: frame_at(video_path, t), wanted)))
    names = {t: f"F{i + 1}" for i, t in enumerate(t for t in wanted if got[t])}
    items = [{"index": i, "claim": text, "frames": [names[t] for t in ts if t in names]}
             for i, (text, ts) in enumerate(claims)]
    items = [x for x in items if x["frames"]]
    if not items:
        return {}
    contents: list[str | bytes] = []
    for t, name in names.items():
        contents += [f"Frame {name} at {_clock(t)}:", got[t]]
    contents.append("CLAIMS (JSON):\n" + json.dumps(items, ensure_ascii=False))
    raw = gemini.generate_json_with_images(contents, list[Verdict], model=config.SC_MODEL_FAST,
                                           system_instruction=SYSTEM, temperature=0.0)
    out: dict[int, Verdict] = {}
    for item in raw if isinstance(raw, list) else []:
        try:
            v = Verdict.model_validate(item)
        except ValidationError:
            continue
        out.setdefault(v.index, v)
    return out


def recompute(likelihood: str, kept: list[str], metadata: list[str]) -> str:
    """Likelihood from the confirmed signs only: never raised by this step."""
    by_count = LIKELIHOOD[min(len(kept), 2)] if len(kept) < 3 else likelihood
    level = min(LIKELIHOOD.index(likelihood), LIKELIHOOD.index(by_count))
    if metadata:
        level = max(level, LIKELIHOOD.index("medium"))
    return LIKELIHOOD[level]


def _visual_claims(inventory: Inventory) -> list[tuple[int, str, list[float]]]:
    """(signal index, text, frame times) of every sign a frame can confirm."""
    syn = inventory.synthetic
    assert syn is not None
    untimed = untimed_frames(inventory)
    return [(i, s, times(s, inventory.duration_s) or untimed) for i, s in enumerate(syn.signals)
            if s not in syn.metadata_signals and not is_audio(s)]


def keep(inventory: Inventory, checked: set[int], confirmed: set[int], *, verified: bool) -> int:
    """Keep the signs that were not checked (audio, file metadata) or were confirmed;
    recompute the likelihood (in place). Returns how many were left out."""
    syn = inventory.synthetic
    assert syn is not None
    kept = [s for i, s in enumerate(syn.signals) if i not in checked or i in confirmed]
    dropped = len(syn.signals) - len(kept)
    meta = set(syn.metadata_signals)
    voice = bool(kept) and all(s not in meta and is_audio(s) for s in kept)
    inventory.synthetic = syn.model_copy(update={
        "signals": kept, "unconfirmed": dropped, "verified": verified,
        "likelihood": recompute(syn.likelihood, kept, syn.metadata_signals) if dropped else syn.likelihood,
        "voice_only": syn.voice_only or voice})
    for seg in inventory.segments:
        seg.synthetic_signals = [x for x in seg.synthetic_signals if is_audio(x)]
    return dropped


def drop_visual(inventory: Inventory) -> None:
    """When the signs cannot be checked at all: show no visual sign (nothing unverified)."""
    if inventory.synthetic is not None and inventory.synthetic.signals:
        keep(inventory, {i for i, _, _ in _visual_claims(inventory)}, set(), verified=False)


def verify(video_path: Path, inventory: Inventory, *, mock: bool | None = None) -> str | None:
    """Check the inventory's signs against frames and keep only what holds (in place).
    Returns an error text for the report when the visual signs could not be checked."""
    mock = config.mock_mode() if mock is None else mock
    syn = inventory.synthetic
    if mock or syn is None or not syn.signals:
        return None
    claims = _visual_claims(inventory)
    error = None
    confirmed: set[int] = set()
    if claims:
        try:
            got = ask(video_path, [(s, ts) for _, s, ts in claims])
            confirmed = {claims[k][0] for k, v in got.items() if 0 <= k < len(claims) and v.visible}
        except gemini.GeminiUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - signs are dropped, and the report says so
            log.warning("AI-sign verification failed: %s", type(e).__name__)
            error = (f"AI-content check: the visual signs could not be verified against the "
                     f"video's frames ({type(e).__name__}), so they are not shown.")
    dropped = keep(inventory, {i for i, _, _ in claims}, confirmed, verified=error is None)
    if claims and error is None:
        events.emit("ai", f"Checked {len(claims)} visual sign{'s' * (len(claims) != 1)} of AI generation "
                    f"against frames of the video: {len(claims) - dropped} confirmed"
                    + (f", {dropped} not confirmed and left out" if dropped else ""), stage="ai")
    return error
