"""TEMPORARY OpenRouter backend, used only until a Gemini AI Studio key exists.

Delete this file and the lines marked `# OPENROUTER-TEMP` (see README,
"Removing OpenRouter"). `app/gemini.py` dispatches here when
`config.provider() == "openrouter"`; the public interface is the same as
`gemini.generate_json` / `gemini.generate_json_with_video`.

FREE MODELS ONLY: every model id must end in ":free" or the call is refused
before anything is sent, and any response reporting a non-zero cost halts all
further calls. The API key is never logged; neither are request bodies.

Gemma 4 on OpenRouter takes text, images and video but no audio, so speech is
transcribed separately by a free audio model and merged into the stage A
prompt. JSON output degrades json_schema -> json_object -> plain prompt with
the schema embedded, then one repair attempt.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import TypeAdapter, ValidationError

from app import config, gemini

log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"
FALLBACK_MODEL = "google/gemma-4-26b-a4b-it:free"
DEFAULT_AUDIO_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

MIN_INTERVAL_S = 3.5
MAX_RETRY_AFTER_S = 60.0
MAX_VIDEO_BYTES = 20 * 1024 * 1024
FRAME_EVERY_S = 2
MAX_FRAMES = 60
TIMEOUT_S = 300.0
# Fall back to the second model on these (404 = model/endpoint unavailable).
FALLBACK_CODES = {404, 429, 500, 502, 503, 504}
MODES = ("json_schema", "json_object", "prompt")

USAGE_LOG = config.STORAGE_DIR / "jobs" / "usage.log"

_transport: httpx.BaseTransport | None = None  # tests install an httpx.MockTransport
_sleep = time.sleep  # patched in tests
_lock = threading.Lock()
_last_call = 0.0
_halted: str | None = None


class PaidModelRefused(ValueError):
    pass


class NonZeroCost(RuntimeError):
    pass


class OpenRouterError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"OpenRouter HTTP {code}: {message}")
        self.code = code  # read by gemini.with_retry
        self.message = message


class BadJSON(ValueError):
    pass


# ---------------------------------------------------------------- config

def _require_free(model: str) -> str:
    if not model.endswith(":free"):
        raise PaidModelRefused(f"refusing non-free OpenRouter model {model!r} (ids must end in ':free')")
    return model


def models() -> list[str]:
    primary = _require_free(os.getenv("SC_OR_MODEL", DEFAULT_MODEL))
    fallback = _require_free(FALLBACK_MODEL)
    return [primary] if primary == fallback else [primary, fallback]


def audio_model() -> str:
    return _require_free(os.getenv("SC_OR_AUDIO_MODEL", DEFAULT_AUDIO_MODEL))


def _key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise gemini.GeminiUnavailable("OPENROUTER_API_KEY is not set (add it to .env)")
    return key


# ---------------------------------------------------------------- HTTP

def _wait_turn() -> None:
    """At most one request every MIN_INTERVAL_S across all threads."""
    global _last_call
    with _lock:
        wait = _last_call + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            _sleep(wait)
        _last_call = time.monotonic()


def _error_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error") or {}
        msg = err.get("message") or ""
        raw = (err.get("metadata") or {}).get("raw")
        if raw:
            msg = f"{msg} ({raw})"
    except (ValueError, AttributeError):
        msg = resp.text
    return str(msg)[:300]


def _log_usage(what: str, model: str, usage: dict[str, Any]) -> None:
    line = json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "what": what, "model": model,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
    })
    log.info("openrouter usage %s", line)
    try:
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        log.warning("could not write %s", USAGE_LOG)


def _post(payload: dict[str, Any], *, what: str) -> str:
    """One chat completion. Returns the message text; raises OpenRouterError."""
    global _halted
    if _halted:
        raise NonZeroCost(_halted)
    _require_free(payload["model"])
    payload = {**payload, "usage": {"include": True}}
    headers = {"Authorization": f"Bearer {_key()}", "X-Title": "StreamClearance"}
    _wait_turn()
    log.info("openrouter %s model=%s format=%s", what, payload["model"],
             (payload.get("response_format") or {}).get("type", "none"))
    with httpx.Client(transport=_transport, timeout=TIMEOUT_S) as client:
        try:
            resp = client.post(API_URL, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            raise OpenRouterError(504, f"timeout ({type(e).__name__})") from None
        except httpx.TransportError as e:
            raise OpenRouterError(503, f"network error ({type(e).__name__})") from None

    if resp.status_code != 200:
        if resp.status_code == 429 or resp.status_code >= 500:
            try:
                retry_after = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                retry_after = 0.0
            if retry_after > 0:
                log.warning("openrouter %s: HTTP %d, Retry-After %.0fs", what, resp.status_code, retry_after)
                _sleep(min(retry_after, MAX_RETRY_AFTER_S))
        raise OpenRouterError(resp.status_code, _error_message(resp))

    body = resp.json()
    usage = body.get("usage") or {}
    _log_usage(what, body.get("model") or payload["model"], usage)
    cost = usage.get("cost") or 0
    if cost:
        _halted = (f"OpenRouter reported a non-zero cost ({cost}) for {payload['model']}; "
                   "stopping all OpenRouter calls. This is a bug: only free models may be used.")
        log.error(_halted)
        raise NonZeroCost(_halted)

    # Provider errors can arrive with HTTP 200.
    if body.get("error"):
        err = body["error"]
        raise OpenRouterError(int(err.get("code") or 502), str(err.get("message", ""))[:300])
    choices = body.get("choices") or []
    if not choices:
        raise OpenRouterError(502, "no choices in response")
    if choices[0].get("error"):
        err = choices[0]["error"]
        raise OpenRouterError(int(err.get("code") or 502), str(err.get("message", ""))[:300])
    return (choices[0].get("message") or {}).get("content") or ""


def _complete(messages: list[dict], *, what: str, extra: dict[str, Any] | None = None,
              model_list: list[str] | None = None) -> str:
    """Primary model, then the free fallback on 429/5xx/unavailable; retried with backoff."""
    model_list = model_list or models()

    def attempt() -> str:
        for i, model in enumerate(model_list):
            try:
                return _post({"model": model, "messages": messages, **(extra or {})}, what=what)
            except OpenRouterError as e:
                if e.code not in FALLBACK_CODES or i == len(model_list) - 1:
                    raise
                log.warning("openrouter %s: %s on %s, trying %s", what, e.code, model, model_list[i + 1])
        raise AssertionError("unreachable")

    return gemini.with_retry(attempt, what=f"openrouter {what}")


# ---------------------------------------------------------------- JSON output

def _schema_name(response_schema: Any) -> str:
    name = getattr(response_schema, "__name__", None) or str(response_schema)
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:60] or "output"


def _is_array(schema: dict) -> bool:
    return schema.get("type") == "array"


def extract_json(text: str) -> Any:
    """First JSON object or array in a model reply (tolerates code fences and prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                return decoder.raw_decode(text, i)[0]
            except json.JSONDecodeError:
                continue
    raise BadJSON("no JSON found in model output")


def _unwrap(value: Any, schema: dict) -> Any:
    """json_object mode can't return a bare array; accept {"items": [...]} for array schemas."""
    if _is_array(schema) and isinstance(value, dict):
        lists = [v for v in value.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return value


def _schema_hint(schema: dict) -> str:
    wrap = (' The JSON must be an object of the form {"items": [...]} where items is the array.'
            if _is_array(schema) else "")
    return (f"\n\nReply with JSON only, matching this JSON Schema:{wrap}\n"
            f"{json.dumps(schema, separators=(',', ':'))}")


def _with_text(parts: list[dict], extra_text: str) -> list[dict]:
    return [*parts, {"type": "text", "text": extra_text}]


def _structured(system: str | None, parts: list[dict], response_schema: Any, *,
                temperature: float, what: str) -> Any:
    """Generate JSON matching response_schema. Returns plain parsed JSON (validated)."""
    adapter = TypeAdapter(response_schema)
    schema = adapter.json_schema()

    def messages(user_parts: list[dict]) -> list[dict]:
        sys = [{"role": "system", "content": system}] if system else []
        return [*sys, {"role": "user", "content": user_parts}]

    def check(text: str) -> Any:
        value = _unwrap(extract_json(text), schema)
        adapter.validate_python(value)
        return value

    last_text, last_error = "", ""
    for mode in MODES:
        extra: dict[str, Any] = {"temperature": temperature}
        user = parts
        if mode == "json_schema":
            extra["response_format"] = {"type": "json_schema", "json_schema": {
                "name": _schema_name(response_schema), "strict": False, "schema": schema}}
        else:
            user = _with_text(parts, _schema_hint(schema))
            if mode == "json_object":
                extra["response_format"] = {"type": "json_object"}
        try:
            last_text = _complete(messages(user), what=f"{what}/{mode}", extra=extra)
            return check(last_text)
        except OpenRouterError as e:
            if e.code != 400:
                raise
            last_error = e.message
            log.warning("openrouter %s: %s rejected (HTTP 400), trying next mode", what, mode)
        except (BadJSON, ValidationError) as e:
            last_error = str(e)[:500]
            log.warning("openrouter %s: %s output invalid (%s), trying next mode",
                        what, mode, last_error.splitlines()[0])

    if not last_text:
        raise OpenRouterError(400, f"every JSON mode was rejected: {last_error}")
    repair = [{"type": "text", "text": (
        "Fix this JSON to match the schema. Keep the content, change only the structure.\n"
        f"Validation error: {last_error}\n\nJSON:\n{last_text[:20000]}{_schema_hint(schema)}")}]
    text = _complete(messages(repair), what=f"{what}/repair", extra={"temperature": 0})
    try:
        return check(text)
    except (BadJSON, ValidationError) as e:
        raise BadJSON(f"OpenRouter output did not match the schema after repair: {str(e)[:300]}") from None


# ---------------------------------------------------------------- public API

def generate_json(prompt: str, response_schema: Any, *, system_instruction: str | None = None,
                  temperature: float = 0.2, **_: Any) -> Any:
    """Same contract as gemini.generate_json (the Gemini model name is ignored)."""
    return _structured(system_instruction, [{"type": "text", "text": prompt}], response_schema,
                       temperature=temperature, what="generate_json")


def generate_json_with_video(video_path: Path, mime_type: str, prompt: str, response_schema: Any, *,
                             system_instruction: str | None = None, temperature: float = 0.2,
                             **_: Any) -> Any:
    """Same contract as gemini.generate_json_with_video, plus "speech_transcribed" in the result.

    The video goes inline as a base64 data URL (transcoded small first when
    ffmpeg exists). If OpenRouter rejects video input, frames every 2 s are
    sent as images instead.
    """
    with tempfile.TemporaryDirectory(prefix="sc-or-") as tmp:
        tmpdir = Path(tmp)
        small, mime = _transcode(video_path, mime_type, tmpdir)
        size = small.stat().st_size
        if size > MAX_VIDEO_BYTES:
            raise ValueError(f"video is {size / 1e6:.1f} MB after transcoding; the OpenRouter "
                             f"bridge accepts at most {MAX_VIDEO_BYTES // (1024 * 1024)} MB")

        transcript = transcribe(video_path, tmpdir)
        if transcript is None:
            speech = ("\n\nNo speech transcript is available; transcribe any speech you can "
                      "infer from captions, otherwise leave transcript empty.")
        else:
            speech = ("\n\nSpeech transcript from a separate speech-to-text pass (use it for "
                      "the transcript fields, matching timestamps):\n" + (transcript or "(no speech)"))
        text = prompt + speech

        b64 = base64.b64encode(small.read_bytes()).decode()
        parts = [{"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}},
                 {"type": "text", "text": text}]
        log.info("openrouter video: %s, %d bytes inline", video_path.name, size)
        try:
            out = _structured(system_instruction, parts, response_schema,
                              temperature=temperature, what="stage_a/video")
        except OpenRouterError as e:
            if e.code in (429,) or e.code >= 500:
                raise
            log.warning("openrouter rejected video input (HTTP %d: %s); falling back to frames "
                        "every %ds + audio", e.code, e.message[:120], FRAME_EVERY_S)
            out = _frames_fallback(video_path, tmpdir, text, system_instruction, response_schema,
                                   temperature)
    if isinstance(out, dict):
        out = {**out, "speech_transcribed": transcript is not None}
    return out


# ---------------------------------------------------------------- media

def ffmpeg_path() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    bundled = sorted(Path.home().glob(".local/tools/ffmpeg-*/bin/ffmpeg"))
    return str(bundled[-1]) if bundled else None


def _ffmpeg(args: list[str]) -> None:
    ff = ffmpeg_path()
    if not ff:
        raise FileNotFoundError("ffmpeg not found")
    subprocess.run([ff, "-v", "error", "-y", *args], check=True, capture_output=True, timeout=600)


def _transcode(src: Path, mime: str, tmpdir: Path) -> tuple[Path, str]:
    """<=480p, low-bitrate H.264 with mono 32k audio. Returns the original if ffmpeg is missing."""
    if not ffmpeg_path():
        log.warning("ffmpeg not found; sending %s untranscoded", src.name)
        return src, mime
    out = tmpdir / "small.mp4"
    _ffmpeg(["-i", str(src), "-vf", "scale=-2:'min(480,ih)',fps=5", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "32", "-maxrate", "300k", "-bufsize", "600k",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", "-b:a", "32k",
             "-movflags", "+faststart", str(out)])
    return out, "video/mp4"


def _extract_audio(src: Path, tmpdir: Path) -> Path | None:
    out = tmpdir / "audio.mp3"
    try:
        _ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
                 "-b:a", "32k", str(out)])
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        log.warning("could not extract audio (%s)", type(e).__name__)
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def _audio_part(mp3: Path) -> dict:
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(mp3.read_bytes()).decode(), "format": "mp3"}}


TRANSCRIBE_PROMPT = """Transcribe all speech in this audio verbatim, in the original language. \
One line per sentence, formatted exactly as:
[start_s-end_s] text
with times in seconds (e.g. [3.0-7.5] Hello there.). Ignore music, tones and noise. \
If there is no speech at all, reply with exactly: NO SPEECH"""


def transcribe(video_path: Path, tmpdir: Path) -> str | None:
    """Timestamped transcript via the free audio model. "" = no speech, None = failed."""
    mp3 = _extract_audio(video_path, tmpdir)
    if mp3 is None:
        return None
    try:
        model = audio_model()
        text = _complete(
            [{"role": "user", "content": [{"type": "text", "text": TRANSCRIBE_PROMPT}, _audio_part(mp3)]}],
            what="transcribe", extra={"temperature": 0}, model_list=[model])
    except NonZeroCost:
        raise
    except Exception as e:  # noqa: BLE001 - speech is optional; carry on without it
        log.warning("speech transcription failed, continuing without it: %s", str(e)[:200])
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.upper().startswith("NO SPEECH"):
        return ""
    lines = [ln.strip() for ln in text.splitlines() if re.match(r"\s*\[\d", ln)]
    return "\n".join(lines) if lines else text[:5000]


def _frames_fallback(video_path: Path, tmpdir: Path, text: str, system: str | None,
                     response_schema: Any, temperature: float) -> Any:
    frames_dir = tmpdir / "frames"
    frames_dir.mkdir()
    _ffmpeg(["-i", str(video_path), "-vf", f"fps=1/{FRAME_EVERY_S},scale=-2:'min(480,ih)'",
             "-frames:v", str(MAX_FRAMES), "-q:v", "6", str(frames_dir / "f%03d.jpg")])
    frames = sorted(frames_dir.glob("f*.jpg"))
    if not frames:
        raise RuntimeError("could not extract frames from the video")
    parts: list[dict] = []
    for i, frame in enumerate(frames):
        parts.append({"type": "text", "text": f"Frame at {i * FRAME_EVERY_S}s:"})
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode()}})
    note = (f"\n\nThe video is given as {len(frames)} frames, one every {FRAME_EVERY_S} seconds "
            "(timestamps above each frame).")
    mp3 = _extract_audio(video_path, tmpdir)
    with_audio = parts + ([_audio_part(mp3)] if mp3 else [])
    log.warning("openrouter stage A: using %d frames%s instead of video",
                len(frames), " + audio" if mp3 else "")
    try:
        return _structured(system, _with_text(with_audio, text + note), response_schema,
                           temperature=temperature, what="stage_a/frames")
    except OpenRouterError as e:
        if not mp3 or e.code == 429 or e.code >= 500:
            raise
        log.warning("openrouter rejected the audio part (HTTP %d); retrying frames only", e.code)
        return _structured(system, _with_text(parts, text + note), response_schema,
                           temperature=temperature, what="stage_a/frames-no-audio")
