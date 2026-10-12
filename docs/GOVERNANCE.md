# Governance: data, retention and AI controls

What StreamClearance does with a video, how long anything is kept, and what
stops the model from making things up. Every row points at the code that makes
it true. Vertex AI's own data handling (training, caching, abuse monitoring,
residency) is in README "Privacy and data handling", with Google's sources.

## Data flow

```
browser ──upload──▶ Cloud Run (asia-southeast1)
                    uploads/<video_id>.<ext> + meta           (container disk)
                         │
                         ├─ ≤ 19 MB: video bytes inline ──▶ Gemini on Vertex AI
                         └─ > 19 MB: gs://<project>-sc-uploads  (asia-southeast1)
                                                         ──▶ Gemini on Vertex AI
                    Gemini location: `global` by default (no residency guarantee),
                    asia-southeast1 with SC_DATA_RESIDENCY=sg
                         │
                    law search: Vertex AI RAG Engine (asia-southeast1), law texts only
                         │
                    ffmpeg (local): one ~320 px JPEG still per flagged moment,
                    taken before the video is deleted
                         │
                    jobs/<job_id>.json  (status + report + activity feed) (container disk)
                    jobs/<job_id>/frames/<start ms>.jpg  (moment stills)  (container disk)
                    jobs/usage.log      (ids, model names, token counts)
                         │
browser ◀──report── GET /api/jobs/<job_id>   (plays its own copy of the video)
```

The video leaves the browser once, for the check. The report holds what the
video shows and says (transcript, on-screen text), the flagged moments and
their legal reasons, one small still image per flagged moment, and the
activity feed (`events`: what the agent watched, searched and decided, which
quotes the video and the model); it never holds the video. Stills and feed
are report data: kept and deleted with the report, never logged. Every `/api` route needs the
access code when `SC_ACCESS_CODE` is set (`app/main.py` `access_code_gate`).

## Retention

| Data | Kept until | Where in code |
|---|---|---|
| Uploaded video, its meta, derived files under `uploads/<id>*` | the check ends (done, failed or cancelled) | `app/jobs.py` `run_job` finally → `app/retention.py` `purge_video` |
| Upload-bucket copy (videos > 19 MB) | the check ends; bucket lifecycle deletes leftovers after 1 day; soft delete off | `app/retention.py` `purge_video` → `app/gemini.py` `delete_gcs_objects` |
| Uploaded video never analysed | `SC_UNUSED_VIDEO_TTL_MIN` (60) minutes | `app/retention.py` `sweep_unused_videos` |
| Report (job record, incl. its activity feed) | `SC_REPORT_TTL_DAYS` (7) days after creation, or "Delete this report now" | `app/retention.py` `expire_reports`, `delete_report`; `DELETE /api/jobs/{id}` |
| Moment stills (`jobs/<id>/frames/`), taken from the video before it is deleted | with the report (same TTL, same delete); folders without a report are swept | `app/frames.py`; `app/storage.py` `delete_job`, `delete_orphan_dirs` |
| "Expired/deleted" marker (id + reason only) | one more TTL, so the UI can say "expired" | `app/storage.py` `sweep_gone` |
| App logs, `jobs/usage.log` | container lifetime; Cloud Logging `_Default` 30 days | content-free: `app/logsafe.py`, `tests/test_logs.py` |
| Cloud Run request logs (URL with ids, IP, time) | 30 days (Cloud Logging `_Default`) | Google Cloud |
| Vertex AI in-memory cache | up to 24 h, unless the project disables it | README "Manual step for zero data retention" |
| Prompts flagged by Google's abuse classifiers | up to 90 days, at Google | Google; exception only on request |

The sweeper runs on startup and every 30 minutes (`app/main.py` `_sweeper`);
reads also expire reports lazily (`GET /api/jobs/{id}`). Each report records
`privacy.video_deleted_at` and `privacy.report_expires_at`.

## AI controls

- **Code verifier, not a model** (`app/brain/verifier.py`): every finding's
  timestamps must lie inside the video, its taxonomy leaf must come from the
  perception stage for an overlapping moment, and every cited provision must
  exist for that country and cover that leaf. Failing citations are removed;
  findings left without one are dropped. The legacy path checks quotes
  against the rule cards (`app/validate.py`).
- **Critic** (`app/brain/agents.py`): a second agent reviews each verified
  finding against the cited provision's text and its exceptions, and may
  downgrade or drop it; the report counts its changes (`critic_changes`).
- **Severity floors** (`app/floors.py`): after the critic, fixed rules in code
  raise (never lower) gambling promotion and serious cure / stop-your-medicine
  claims to red when a cited in-force provision binding the publisher
  genuinely matches; each raise is noted on the finding and in the feed.
  Judgement calls run at temperature 0 with a fixed seed.
- **AI-generated content check** (`app/synthetic.py`): the model only
  reports evidence (signs, label, real person); a fixed table in code decides
  each country's consequence and cites only provisions listed there (tests
  check each quote on its page). Wording is "signs of AI generation".
- **No silent fallback, no hang**: a simplified assessor check is recorded
  (`method`, `fallback_reason`, feed warning, UI tag); agent runs and Gemini
  requests have deadlines (`SC_AGENT_TIMEOUT_S`, `SC_GEMINI_HTTP_TIMEOUT_S`).
- **Activity feed** (`app/events.py`): only real pipeline steps; report data,
  never logged.
- **Draft / reviewed labels**: every provision carries `review.status`; a law
  is "reviewed" only when a person signed off all its provisions. The UI and
  the report show draft laws as draft (`rules_status`, provenance `status`).
- **Call cap**: `SC_MAX_CALLS_PER_JOB` (40) model calls per job, direct and
  ADK, then the job stops with an error (`app/gemini.py` `job_budget`).
- **Provenance** (`app/provenance.py`, report `provenance`): created_at, app
  git sha, brain, model ids and Vertex location, and per law: id, title,
  draft/reviewed, source PDF sha256, provision count and the date the
  provisions file was last committed. Shown as "How this report was made".
- **Human decision**: results say "Automated pre-check, not legal advice. A
  person makes the publish decision."; nothing is published or blocked by
  the app.

## Known limits

- Vertex AI's 24-hour in-memory cache is on until a project admin disables it
  (the app's service account lacks `aiplatform.cacheConfigs.update`), and
  abuse-monitoring logging needs an exception from Google, so there is no
  zero data retention yet.
- The default Gemini location `global` gives no data residency guarantee;
  `SC_DATA_RESIDENCY=sg` keeps it in Singapore with the weaker
  `gemini-2.5-flash`.
- Storage is the Cloud Run container's disk: with `--max-instances 1` that is
  one copy, but reports also vanish when the instance is replaced (before
  their TTL). A shared store (GCS/Firestore) would need the same deletion
  rules.
- Most provisions are drafts, not reviewed by a lawyer. Countries without
  provisions use rule cards (basic coverage), some generated as dry runs.
- Model output can still be wrong in ways the verifier can't catch (a missed
  moment, a misread line); the critic is itself a model.
- The browser keeps the file name and ids in localStorage, and its own copy of
  the video until the page is reloaded.
- Not a compliance certification: designed with Malaysia PDPA 2010 and India
  DPDP Act 2023 principles in mind (data minimisation, purpose limitation,
  retention limits).
