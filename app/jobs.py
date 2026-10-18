"""Analysis jobs: created by POST /api/analyze, run as background asyncio tasks.

Status is persisted to storage after every change so GET /api/jobs/{id} (on
any instance, once storage is shared) sees progress. Stages:
uploaded -> inventory -> country:XX (one per country) -> translation -> done.

When a job ends (done, error or cancelled) its video is deleted (app/retention.py);
the report keeps only what the UI needs, so it works without the video.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any

from app import ai_verify, config, events, frames, gemini, metrics, pipeline, provenance, retention
from app.schemas import CountryReport, Inventory, Privacy, Report
from app.storage import get_storage

log = logging.getLogger(__name__)

# Running jobs: job_id -> (task, video_id). Also keeps task references alive.
_active: dict[str, tuple[asyncio.Task, str]] = {}


def active_video_ids() -> set[str]:
    return {video_id for _, video_id in _active.values()}


def active_job_ids() -> set[str]:
    return set(_active)


def _stage(key: str, label: str, status: str = "pending", detail: str = "") -> dict[str, str]:
    return {"key": key, "label": label, "status": status, "detail": detail}


def create_job(video_id: str, countries: list[str]) -> dict[str, Any]:
    now = time.time()
    job = {
        "job_id": uuid.uuid4().hex,
        "video_id": video_id,
        "countries": countries,
        "status": "queued",
        "mock": config.mock_mode(),
        "created_at": now,
        "updated_at": now,
        "error": None,
        "stages": [
            _stage("uploaded", "Video uploaded", "done"),
            _stage("inventory", "Inventory (what's in the video)"),
            *[_stage(f"country:{cc}", f"{config.COUNTRIES[cc].name} rules") for cc in countries],
            _stage("translation", "Translation drift check"),
            _stage("done", "Report ready"),
        ],
        "report": None,
        "events": [],  # the activity feed (app/events.py)
        "video_deleted_at": None,
        "expires_at": None,
    }
    job["expires_at"] = retention.iso(expires_at(job))
    _save(job)
    return job


def start_job(job: dict[str, Any]) -> None:
    job_id = job["job_id"]
    task = asyncio.create_task(run_job(job))
    _active[job_id] = (task, job["video_id"])
    task.add_done_callback(lambda _: _active.pop(job_id, None))


def expires_at(job: dict[str, Any]) -> float:
    return job["created_at"] + config.report_ttl_days() * 86400


def _save(job: dict[str, Any]) -> None:
    store = get_storage()
    if store.gone_reason(job["job_id"]):
        return  # deleted while running: never write it back
    job["updated_at"] = time.time()
    meter = metrics.current()
    if meter is not None and meter.job_id == job["job_id"]:
        job["timings"], job["calls"] = meter.snapshot()
    feed = events.current()
    if feed is not None and feed.job_id == job["job_id"]:
        job["events"] = feed.snapshot()
        _saved_events[job["job_id"]] = feed.version
    store.save_job(job["job_id"], job)


async def delete_job(job_id: str) -> None:
    """Stop the job if it is running, then delete it and everything stored for it."""
    if job_id in _active:
        task, _ = _active[job_id]
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
            await asyncio.wait_for(asyncio.shield(task), timeout=10)
    await asyncio.to_thread(retention.delete_report, job_id, "deleted")


async def _flush_events(job: dict[str, Any], feed: events.EventLog) -> None:
    """Save the job when the feed changed, so events show up while a stage runs."""
    while True:
        await asyncio.sleep(EVENT_FLUSH_S)
        if feed.version != _saved_events.get(job["job_id"]):
            _save(job)


EVENT_FLUSH_S = 1.0
_saved_events: dict[str, int] = {}  # job id -> feed version last saved


def _set(job: dict[str, Any], key: str, status: str, detail: str = "") -> None:
    for st in job["stages"]:
        if st["key"] == key:
            st["status"], st["detail"] = status, detail
    _save(job)


async def run_job(job: dict[str, Any]) -> None:
    store = get_storage()
    mock = job["mock"]
    gemini.job_budget(job["job_id"])  # cost guard: SC_MAX_CALLS_PER_JOB model calls per job
    meter = metrics.start(job["job_id"])  # per-stage timings and calls -> job["timings"/"calls"]
    feed = events.start(job["job_id"], job["created_at"])  # the activity feed -> job["events"]
    flusher = asyncio.create_task(_flush_events(job, feed))
    t_job = time.perf_counter()
    drift_task: asyncio.Task | None = None
    frames_task: asyncio.Task | None = None
    ai_task: asyncio.Task | None = None  # verifies stage A's signs of AI generation
    seen_inventory: Inventory | None = None  # set once stage A is done
    job["status"] = "running"
    current = "inventory"
    try:
        meta = store.load_video_meta(job["video_id"])
        path = store.video_path(job["video_id"])
        if meta is None or path is None:
            raise FileNotFoundError("uploaded video not found")
        if meta.get("probe_s") is not None:
            meter.add_time("upload_probe", meta["probe_s"])

        async def on_start(cc: str) -> None:
            _set(job, f"country:{cc}", "running")

        async def on_done(cc: str, rep: CountryReport | None, err: str | None) -> None:
            if rep is None:
                events.emit("warn", f"{config.COUNTRIES[cc].name}: the check failed ({err or 'error'}).",
                            stage="country", country=cc)
                _set(job, f"country:{cc}", "error", err or "failed")
            else:
                if ai_task is not None:
                    await asyncio.shield(ai_task)  # the AI-content check uses verified signs only
                pipeline.finalize(rep, seen_inventory)
                n = len(rep.findings)
                pipeline.country_done_event(rep)
                _set(job, f"country:{cc}", "done",
                     f"{rep.badge} · {n} finding{'s' * (n != 1)}")

        # The drift check needs only the inventory: start it as soon as stage A is
        # done, so it runs while the countries are assessed instead of after them.
        async def drift(inv: Inventory):
            with metrics.timed("translation"):
                return await pipeline.run_drift(inv, job["countries"], mock=mock)

        async def verify_ai(inv: Inventory) -> str | None:
            """Stage A's visual signs of AI generation, checked against frames before
            anything uses or shows them (app/ai_verify.py); runs alongside the countries."""
            try:
                with metrics.stage("ai_verify"):
                    error = await asyncio.to_thread(ai_verify.verify, path, inv, mock=mock)
            except Exception as e:  # noqa: BLE001 - e.g. the call budget: show no visual sign
                log.error("AI-sign verification failed: %s", type(e).__name__)
                ai_verify.drop_visual(inv)
                error = f"AI-content check: the visual signs could not be verified ({type(e).__name__})."
            pipeline.synthetic_event(inv)
            return error

        async def on_inventory(inv: Inventory) -> None:
            nonlocal drift_task, frames_task, ai_task, seen_inventory
            seen_inventory = inv
            pipeline.inventory_events(inv, ai=False)
            ai_task = asyncio.create_task(verify_ai(inv))
            # Thumbnails come from the video, so they are taken now (in parallel), before
            # the video is deleted at the end of the job.
            frames_task = asyncio.create_task(asyncio.to_thread(frames.extract, path, job["job_id"], inv))
            if any(config.COUNTRIES[cc].drift_languages for cc in job["countries"]) \
                    and pipeline.has_speech(inv):
                _set(job, "translation", "running")
                drift_task = asyncio.create_task(drift(inv))

        if config.brain() == "adk":
            from app import brain

            async def on_stage(key: str, status: str, detail: str) -> None:
                nonlocal current
                current = "countries" if (key, status) == ("inventory", "done") else current
                _set(job, key, status, detail)

            inventory, results = await brain.run(
                path, meta.get("duration_s"), job["countries"], mock=mock,
                on_stage=on_stage, on_start=on_start, on_done=on_done, on_inventory=on_inventory)
        else:
            _set(job, "inventory", "running")
            if mock:
                await asyncio.sleep(config.mock_delay_s())
            with metrics.stage("stage_a"):
                inventory = await asyncio.to_thread(pipeline.stage_a, path, meta.get("duration_s"),
                                                    mock=mock)
            _set(job, "inventory", "done", f"{len(inventory.segments)} segments")
            current = "countries"
            await on_inventory(inventory)
            results = await pipeline.run_countries(
                inventory, job["countries"], mock=mock, on_start=on_start, on_done=on_done)
        reports = {cc: r for cc, r in results.items() if isinstance(r, CountryReport)}
        errors = [f"{cc}: {pipeline.public_error(r)}" for cc, r in results.items()
                  if isinstance(r, Exception)]
        if not reports:
            raise RuntimeError("every country failed: " + "; ".join(errors))

        current = "translation"
        if not any(config.COUNTRIES[cc].drift_languages for cc in reports):
            _set(job, "translation", "skipped", "English-speaking audiences only")
        elif drift_task is None:  # no speech
            pipeline.attach_drift(reports, {}, {cc: pipeline.NO_SPEECH for cc in reports
                                                if config.COUNTRIES[cc].drift_languages})
            _set(job, "translation", "skipped", "no speech to translate")
        else:
            checks, notes, t_errors = await drift_task
            pipeline.attach_drift(reports, checks, notes)
            errors += t_errors
            n = sum(len(r.translation) for r in reports.values())
            _set(job, "translation", "error" if t_errors and not n else "done",
                 f"{n} line checks" + (f", {len(t_errors)} failed" if t_errors else ""))

        if ai_task is not None and (ai_error := await ai_task):
            errors.append(ai_error)
        made = await frames_task if frames_task is not None else {}
        frames.attach(inventory, made)
        if made:
            events.emit("watch", f"Took {len(made)} still frame{'s' * (len(made) != 1)} of the flagged "
                        "moments as visual evidence", stage="report")
        report = Report(
            video_id=job["video_id"],
            duration_s=inventory.duration_s,
            countries=[reports[cc] for cc in job["countries"] if cc in reports],
            mock=mock,
            inventory=inventory,
            errors=errors,
            privacy=Privacy(report_expires_at=retention.iso(expires_at(job))),
            provenance=await asyncio.to_thread(
                provenance.provenance, [cc for cc in job["countries"] if cc in reports], mock,
                job["created_at"]),
        )
        job["report"] = report.model_dump(mode="json")
        checked = sum(c.overall_risk != "not_checked" for c in report.countries)
        events.emit("info", f"Report ready: {checked} of {len(job['countries'])} countries checked"
                    + (f", {len(errors)} check(s) failed" if errors else "") + ".", stage="report")
        job["status"] = "done"
    except asyncio.CancelledError:
        log.warning("job %s cancelled", job["job_id"])
        job["status"] = "cancelled"
        job["error"] = "The analysis was cancelled."
        raise
    except Exception as e:  # noqa: BLE001 - surfaced to the client
        log.error("job %s failed: %s", job["job_id"], type(e).__name__)
        job["status"] = "error"
        job["error"] = pipeline.public_error(e)
        for st in job["stages"]:
            if st["status"] == "running" or (current == "inventory" and st["key"] == "inventory"):
                st["status"], st["detail"] = "error", job["error"]
    finally:
        flusher.cancel()
        _saved_events.pop(job["job_id"], None)
        if drift_task is not None and not drift_task.done():
            drift_task.cancel()
        if ai_task is not None and not ai_task.done():
            ai_task.cancel()
        if frames_task is not None and not frames_task.done():
            # ffmpeg reads the video: let it finish (it has its own timeout) before deleting it
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(frames_task), timeout=60)
        # The video is not needed any more, whatever happened: delete it before
        # the client sees the final status.
        deleted_at = retention.purge_video(job["video_id"])
        job["video_deleted_at"] = deleted_at
        if job["report"] is not None:
            job["report"]["privacy"]["video_deleted_at"] = deleted_at
        if job["status"] == "running":
            job["status"] = "error"
        meter.add_time("total", time.perf_counter() - t_job)
        timing_event(meter)
        _set(job, "done", "done" if job["status"] == "done" else "error")


def timing_event(meter: metrics.Meter) -> None:
    """Where the time went (also in job["timings"]): stage A, the slowest country, translation."""
    t, _ = meter.snapshot()
    countries = {k.split(":", 1)[1]: v for k, v in t.items() if k.startswith("assessor:")}
    for k, v in t.items():
        if k.startswith("critic:"):
            countries[k.split(":", 1)[1]] = countries.get(k.split(":", 1)[1], 0) + v
    parts = [f"total {t.get('total', 0):.0f}s", f"video inventory {t.get('stage_a', 0):.0f}s"]
    if countries:
        slow = max(countries, key=countries.get)
        parts.append(f"slowest country {config.COUNTRIES[slow].name} {countries[slow]:.0f}s")
    if "translation" in t:
        parts.append(f"translation {t['translation']:.0f}s (alongside the countries)")
    events.emit("info", "Timings: " + " · ".join(parts), stage="report")
