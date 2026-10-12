"""The single entry point for every Gemini call: Vertex AI or an AI Studio key.

Backend (see `config.gemini_backend`): Vertex AI when a GCP project is set
(credentials from ADC: GOOGLE_APPLICATION_CREDENTIALS locally, the attached
service account on Cloud Run), else AI Studio when GEMINI_API_KEY is set.

Tests mock `generate_json` / `generate_json_with_video`. Keys are read from the
environment and never printed or logged. Calls are blocking; the pipeline runs
them in worker threads. Rate-limit (429) and transient 5xx errors are retried
with exponential backoff. Token usage of every call goes to jobs/usage.log, and
`job_budget` caps the number of calls per job (SC_MAX_CALLS_PER_JOB).
"""
from __future__ import annotations

import contextvars
import json
import logging
import random
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, TypeVar

from app import config, metrics

log = logging.getLogger(__name__)

T = TypeVar("T")

RETRY_CODES = {429, 500, 503}
FILE_ACTIVE_TIMEOUT_S = 300


class GeminiUnavailable(RuntimeError):
    pass


class CallBudgetExceeded(GeminiUnavailable):
    pass


_client = None
_client_backend: str | None = None
_sleep = time.sleep  # patched in tests

USAGE_LOG = config.STORAGE_DIR / "jobs" / "usage.log"
_usage_lock = threading.Lock()


def _get_client():
    global _client, _client_backend
    backend = config.gemini_backend()
    if _client is None or _client_backend != backend:
        from google import genai

        if backend == "vertex":
            project = config.gcp_project()
            if not project:
                raise GeminiUnavailable("SC_GEMINI_BACKEND=vertex needs GOOGLE_CLOUD_PROJECT")
            _client = genai.Client(vertexai=True, project=project, location=config.vertex_location())
        elif backend == "studio":
            _client = genai.Client(api_key=config.gemini_api_key())
        else:
            raise GeminiUnavailable(
                "No Gemini backend: set GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY")
        _client_backend = backend
    return _client


# ---------------------------------------------------------------- cost guard + usage log

class _Budget:
    def __init__(self, job_id: str, limit: int):
        self.job_id, self.limit, self.used = job_id, limit, 0
        self._lock = threading.Lock()

    def take(self, what: str) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise CallBudgetExceeded(
                    f"Model call limit reached ({self.limit} per job, SC_MAX_CALLS_PER_JOB); "
                    f"refused {what}")
            self.used += 1


_budget: contextvars.ContextVar[_Budget | None] = contextvars.ContextVar("gemini_budget", default=None)


def job_budget(job_id: str, limit: int | None = None) -> _Budget:
    """Start counting calls for this job in the current context (inherited by
    asyncio tasks and asyncio.to_thread workers)."""
    b = _Budget(job_id, config.MAX_CALLS_PER_JOB if limit is None else limit)
    _budget.set(b)
    return b


def count_call(what: str) -> None:
    """Raise CallBudgetExceeded if the current job has used up its calls."""
    b = _budget.get()
    if b is not None:
        b.take(what)


def log_usage(what: str, model: str, usage: Any, *, stage: str | None = None,
              seconds: float | None = None) -> None:
    """Append one call's token counts to jobs/usage.log (usage = genai usage_metadata),
    and count it in the job's metrics under `stage` (default: the current stage)."""
    b = _budget.get()
    tokens = {k: getattr(usage, f"{k}_token_count", None) if usage is not None else None
              for k in ("prompt", "candidates", "thoughts", "total")}
    metrics.record_call(stage, tokens, seconds)
    line = json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "job": b.job_id if b else None,
                       "what": what, "model": model, **tokens})
    log.info("gemini usage %s", line)
    try:
        with _usage_lock:
            USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with USAGE_LOG.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        log.warning("could not write %s", USAGE_LOG)


def _generate(client, model: str, contents: Any, cfg: Any, *, what: str,
              hedge_after: float | None = None):
    count_call(what)
    t0 = time.perf_counter()

    def call():
        return with_retry(
            lambda: client.models.generate_content(model=model, contents=contents, config=cfg),
            what=what)

    resp = _hedged(call, hedge_after, what) if hedge_after else call()
    log_usage(what, getattr(resp, "model_version", None) or model,
              getattr(resp, "usage_metadata", None), seconds=time.perf_counter() - t0)
    return resp


def _hedged(call: Callable[[], T], after: float, what: str) -> T:
    """Run `call`; if it has not answered after `after` seconds, send the same request
    again and take whichever answers first. Single calls sometimes take 4-5x their
    usual time (measured: stage A 10-15s usually, 40-70s in about a third of runs), and
    the slowest call sets the job's time. The duplicate counts against the job's call
    budget; when the budget is used up, no duplicate is sent."""
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini-hedge")
    try:
        first = pool.submit(contextvars.copy_context().run, call)
        done, _ = wait([first], timeout=after)
        if done:
            return first.result()
        try:
            count_call(f"{what} (hedge)")
        except CallBudgetExceeded:
            return first.result()
        log.warning("gemini %s: no answer after %.0fs, sending a hedge request", what, after)
        metrics.record_hedge()
        pending = {first, pool.submit(contextvars.copy_context().run, call)}
        error: BaseException | None = None
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                if f.exception() is None:
                    return f.result()
                error = f.exception()
        assert error is not None
        raise error
    finally:
        pool.shutdown(wait=False)  # the losing request finishes in the background; its answer is dropped


def iter_json(text: str):
    """Every top-level JSON object or array in a model reply, in order (tolerates
    code fences and prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] in "[{":
            try:
                value, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            yield value
            i = end
        else:
            i += 1


def extract_json(text: str) -> Any:
    """First JSON object or array in a model reply."""
    for value in iter_json(text):
        return value
    raise ValueError("no JSON found in model output")


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    text = str(exc)
    return 429 if "RESOURCE_EXHAUSTED" in text or "429" in text[:40] else None


def with_retry(fn: Callable[[], T], *, what: str, retries: int | None = None) -> T:
    """Run fn, retrying 429/5xx with exponential backoff and jitter."""
    retries = config.GEMINI_MAX_RETRIES if retries is None else retries
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - the SDK raises several types
            code = _status_code(e)
            if code not in RETRY_CODES or attempt == retries:
                raise
            delay = min(60.0, 2.0 * 2**attempt) + random.uniform(0, 1)
            log.warning("gemini %s: HTTP %s, retry %d/%d in %.1fs",
                        what, code, attempt + 1, retries, delay)
            _sleep(delay)
    raise AssertionError("unreachable")


def _config(response_schema: Any, system_instruction: str | None, temperature: float,
            thinking: str | None = None, media_resolution: str | None = None):
    """thinking: "minimal"/"low"/"medium"/"high" (Gemini 3 thinking level; None = model
    default). media_resolution: "low"/"medium"/"high" for video frames (None = default)."""
    from google.genai import types

    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        system_instruction=system_instruction,
        temperature=temperature,
        seed=config.MODEL_SEED,
        thinking_config=types.ThinkingConfig(thinking_level=thinking.upper()) if thinking else None,
        media_resolution=f"MEDIA_RESOLUTION_{media_resolution.upper()}" if media_resolution else None,
    )


def _parse(resp: Any, model: str) -> Any:
    if not resp.text:
        raise RuntimeError(f"Empty Gemini response (model={model})")
    return json.loads(resp.text)


def generate_json(
    prompt: str,
    response_schema: Any,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    thinking: str | None = None,
    hedge_after: float | None = None,
) -> Any:
    """Call Gemini with structured JSON output and return the parsed JSON. hedge_after:
    seconds before a duplicate request is sent (default SC_HEDGE_AFTER_S; see
    config.text_hedge_after for calls about a video)."""
    model = model or config.SC_MODEL_SMART
    cfg = _config(response_schema, system_instruction, temperature, thinking)
    log.info("gemini.generate_json model=%s prompt_chars=%d", model, len(prompt))
    resp = _generate(_get_client(), model, prompt, cfg, what="generate_json",
                     hedge_after=hedge_after if hedge_after is not None else config.HEDGE_AFTER_S)
    return _parse(resp, model)


def generate_json_with_video(
    video_path: Path,
    mime_type: str,
    prompt: str,
    response_schema: Any,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    thinking: str | None = None,
    media_resolution: str | None = None,
    hedge_after: float | None = None,
) -> Any:
    """One multimodal call over the whole video (frames and audio) -> JSON.

    Vertex: inline bytes up to SC_VERTEX_INLINE_MB, else a temporary object in
    the GCS upload bucket. AI Studio: the Files API. Uploads are deleted
    afterwards (best effort; the bucket and AI Studio also expire them).
    """
    model = model or config.SC_MODEL_FAST
    client = _get_client()
    cfg = _config(response_schema, system_instruction, temperature, thinking, media_resolution)
    if config.gemini_backend() == "vertex":
        size = video_path.stat().st_size
        if size <= config.VERTEX_INLINE_MAX_BYTES:
            from google.genai import types

            log.info("gemini.generate_json_with_video model=%s inline %d bytes", model, size)
            part = types.Part.from_bytes(data=video_path.read_bytes(), mime_type=mime_type)
            return _parse(_generate(client, model, [part, prompt], cfg,
                                    what="generate_json_with_video", hedge_after=hedge_after), model)
        return _video_via_gcs(client, video_path, mime_type, prompt, cfg, model, hedge_after)
    return _video_via_files_api(client, video_path, mime_type, prompt, cfg, model, hedge_after)


def _video_via_files_api(client, video_path: Path, mime_type: str, prompt: str, cfg, model: str,
                         hedge_after: float | None = None) -> Any:
    log.info("gemini.files.upload %s (%d bytes)", video_path.name, video_path.stat().st_size)
    f = with_retry(
        lambda: client.files.upload(file=str(video_path), config={"mime_type": mime_type}),
        what="files.upload",
    )
    try:
        deadline = time.monotonic() + FILE_ACTIVE_TIMEOUT_S
        while (state := getattr(f.state, "name", str(f.state))) == "PROCESSING":
            if time.monotonic() > deadline:
                raise RuntimeError("Gemini file processing timed out")
            _sleep(2)
            f = with_retry(lambda: client.files.get(name=f.name), what="files.get")
        if state != "ACTIVE":
            raise RuntimeError(f"Gemini file processing failed (state={state})")
        log.info("gemini.generate_json_with_video model=%s", model)
        return _parse(_generate(client, model, [f, prompt], cfg, what="generate_json_with_video",
                                hedge_after=hedge_after), model)
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:  # noqa: BLE001
            log.warning("could not delete Gemini file %s", f.name)


GCS_UPLOAD_URL = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
GCS_OBJECT_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o/{name}"
GCS_LIST_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"


def _gcs_session():
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_write"])
    return AuthorizedSession(creds)


def gcs_prefix(video_id: str) -> str:
    return f"videos/{video_id}/"


def delete_gcs_objects(video_id: str) -> int:
    """Delete every upload-bucket object stored for this video. Returns how many."""
    from urllib.parse import quote

    bucket = config.upload_bucket()
    session = _gcs_session()
    r = session.get(GCS_LIST_URL.format(bucket=bucket),
                    params={"prefix": gcs_prefix(video_id), "fields": "items(name)"}, timeout=60)
    if r.status_code == 404:
        return 0
    if r.status_code >= 300:
        raise RuntimeError(f"GCS list failed: HTTP {r.status_code}")
    n = 0
    for item in r.json().get("items", []):
        d = session.delete(GCS_OBJECT_URL.format(bucket=bucket, name=quote(item["name"], safe="")),
                           timeout=60)
        if d.status_code < 300 or d.status_code == 404:
            n += 1
        else:
            raise RuntimeError(f"GCS delete failed: HTTP {d.status_code}")
    return n


def _video_via_gcs(client, video_path: Path, mime_type: str, prompt: str, cfg, model: str,
                   hedge_after: float | None = None) -> Any:
    from urllib.parse import quote

    from google.genai import types

    bucket = config.upload_bucket()
    # Under the video id, so app/retention.py can find and delete it when the job ends.
    name = f"{gcs_prefix(video_path.stem)}{uuid.uuid4().hex}{video_path.suffix.lower()}"
    session = _gcs_session()
    log.info("gcs upload %s -> gs://%s/%s (%d bytes)", video_path.name, bucket, name,
             video_path.stat().st_size)
    with video_path.open("rb") as fh:
        r = session.post(GCS_UPLOAD_URL.format(bucket=bucket),
                         params={"uploadType": "media", "name": name},
                         headers={"Content-Type": mime_type}, data=fh, timeout=600)
    if r.status_code >= 300:
        raise RuntimeError(f"GCS upload failed: HTTP {r.status_code}")
    try:
        part = types.Part.from_uri(file_uri=f"gs://{bucket}/{name}", mime_type=mime_type)
        log.info("gemini.generate_json_with_video model=%s gcs", model)
        return _parse(_generate(client, model, [part, prompt], cfg, what="generate_json_with_video",
                                hedge_after=hedge_after), model)
    finally:
        try:
            session.delete(GCS_OBJECT_URL.format(bucket=bucket, name=quote(name, safe="")), timeout=60)
        except Exception:  # noqa: BLE001
            log.warning("could not delete gs://%s/%s", bucket, name)
