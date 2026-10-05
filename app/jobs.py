"""Analysis jobs: created by POST /api/analyze, run as background asyncio tasks.

Status is persisted to storage after every change so GET /api/jobs/{id} (on
any instance, once storage is shared) sees progress. Stages:
uploaded -> inventory -> country:XX (one per country) -> translation -> done.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from app import config, gemini, pipeline
from app.schemas import CountryReport, Report
from app.storage import get_storage

log = logging.getLogger(__name__)

_tasks: set[asyncio.Task] = set()  # keep references so tasks aren't garbage-collected


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
    }
    _save(job)
    return job


def start_job(job: dict[str, Any]) -> None:
    task = asyncio.create_task(run_job(job))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def _save(job: dict[str, Any]) -> None:
    job["updated_at"] = time.time()
    get_storage().save_job(job["job_id"], job)


def _set(job: dict[str, Any], key: str, status: str, detail: str = "") -> None:
    for st in job["stages"]:
        if st["key"] == key:
            st["status"], st["detail"] = status, detail
    _save(job)


async def run_job(job: dict[str, Any]) -> None:
    store = get_storage()
    mock = job["mock"]
    gemini.job_budget(job["job_id"])  # cost guard: SC_MAX_CALLS_PER_JOB model calls per job
    job["status"] = "running"
    current = "inventory"
    try:
        meta = store.load_video_meta(job["video_id"])
        path = store.video_path(job["video_id"])
        if meta is None or path is None:
            raise FileNotFoundError("uploaded video not found")

        async def on_start(cc: str) -> None:
            _set(job, f"country:{cc}", "running")

        async def on_done(cc: str, rep: CountryReport | None, err: str | None) -> None:
            if rep is None:
                _set(job, f"country:{cc}", "error", err or "failed")
            else:
                n = len(rep.findings)
                _set(job, f"country:{cc}", "done",
                     f"{rep.badge} · {n} finding{'s' * (n != 1)}")

        if config.brain() == "adk":
            from app import brain

            async def on_stage(key: str, status: str, detail: str) -> None:
                nonlocal current
                current = "countries" if (key, status) == ("inventory", "done") else current
                _set(job, key, status, detail)

            inventory, results = await brain.run(
                path, meta.get("duration_s"), job["countries"], mock=mock,
                on_stage=on_stage, on_start=on_start, on_done=on_done)
        else:
            _set(job, "inventory", "running")
            if mock:
                await asyncio.sleep(config.mock_delay_s())
            inventory = await asyncio.to_thread(pipeline.stage_a, path, meta.get("duration_s"), mock=mock)
            _set(job, "inventory", "done", f"{len(inventory.segments)} segments")
            current = "countries"
            results = await pipeline.run_countries(
                inventory, job["countries"], mock=mock, on_start=on_start, on_done=on_done)
        reports = {cc: r for cc, r in results.items() if isinstance(r, CountryReport)}
        errors = [f"{cc}: {pipeline.public_error(r)}" for cc, r in results.items()
                  if isinstance(r, Exception)]
        if not reports:
            raise RuntimeError("every country failed: " + "; ".join(errors))

        current = "translation"
        if any(config.COUNTRIES[cc].drift_languages and r.findings for cc, r in reports.items()):
            _set(job, "translation", "running")
            t_errors = await pipeline.run_translation(inventory, reports, mock=mock)
            errors += t_errors
            checks = sum(len(r.translation) for r in reports.values())
            _set(job, "translation", "error" if t_errors and not checks else "done",
                 f"{checks} line checks" + (f", {len(t_errors)} failed" if t_errors else ""))
        else:
            _set(job, "translation", "skipped", "no flagged speech for non-English audiences")

        report = Report(
            video_id=job["video_id"],
            duration_s=inventory.duration_s,
            countries=[reports[cc] for cc in job["countries"] if cc in reports],
            mock=mock,
            inventory=inventory,
            errors=errors,
        )
        job["report"] = report.model_dump(mode="json")
        job["status"] = "done"
        _set(job, "done", "done")
    except Exception as e:  # noqa: BLE001 - surfaced to the client
        log.exception("job %s failed", job["job_id"])
        job["status"] = "error"
        job["error"] = pipeline.public_error(e)
        for st in job["stages"]:
            if st["status"] == "running" or (current == "inventory" and st["key"] == "inventory"):
                st["status"], st["detail"] = "error", job["error"]
        _set(job, "done", "error")
