# StreamClearance

Upload a short video, pick APAC countries, get a per-country risk report with
timestamped flags tied to the exact law section and page. Advisory only, not
legal advice. Most of these laws put duties on platforms, so the practical risk
is takedown or geo-blocking in that country.

## Status: M0–M5 working; Gemini on Vertex AI; deployable to Cloud Run

| Path | What it does |
|---|---|
| `app/config.py` | Paths, model IDs (`SC_MODEL_FAST`, `SC_MODEL_SMART`), country → law map |
| `app/schemas.py` | Pydantic models: `RuleCard`, `Segment`, `Inventory`, `Finding`, `CountryReport`, `Report`, `Category` |
| `app/gemini.py` | The only module that calls Gemini directly (google-genai: Vertex AI or AI Studio key); usage log + per-job call cap |
| `app/validate.py` | `validate_findings()` drops findings with unknown card IDs or out-of-range timestamps |
| `app/storage.py` | Local `uploads/` and `jobs/` behind a small interface (swap for GCS later) |
| `app/media.py` | ffprobe duration (optional; skipped if ffprobe is missing) |
| `app/pipeline.py` | Stage A inventory, stage B per-country findings, translation-drift check, mocks |
| `app/jobs.py` | Background analysis jobs, status persisted to `jobs/<id>.json` |
| `app/main.py` | FastAPI routes + optional access code + static UI |
| `app/static/` | Plain HTML/CSS/JS UI (no framework, no CDN) |
| `tools/extract_text.py` | `data/laws/raw/*.pdf` → `data/laws/text/*.json` (`[{page, text}]`) |
| `app/law_search.py` | Vertex AI RAG Engine client (law search, retrieval path B) |
| `tools/index_laws.py` | Sync provisions + full law text into the RAG Engine corpora → `data/search_index.json` |
| `tools/build_rules.py` | Law text → Gemini-drafted rule cards → quote/page check → `data/rules/<cc>.json` |

Countries: IN (IT Rules 2021, as updated Feb 2026), AU (Online Safety Act 2021),
MY (Online Safety Act 2025), SG (Online Safety (Relief and Accountability) Bill
2025, **the bill as introduced, so check it against the enacted Act**), PK (PECA Amendment 2025).

Deployment: see [Cloud Run](#cloud-run) below.

## Running the app

```bash
./run.sh                      # http://localhost:8430  (SC_PORT to change)
```

Without a Gemini backend (or with `SC_MOCK=1`) everything runs in **mock mode**:
stage A returns a canned inventory scaled to the real video length, stage B
returns canned findings (plus one invented card ID the validator must drop),
and the drift check returns one canned "increased" example. The UI shows a
MOCK MODE banner. See [Gemini backends](#gemini-backends) to use Gemini.

If `data/rules/<cc>.json` doesn't exist, the app falls back to
`<cc>.dryrun.json`, generating it with the `build_rules --dry-run` logic, and
marks that country's report `draft/unreviewed`.

API: `POST /api/upload` (multipart `file`), `POST /api/analyze
{video_id, countries[]}`, `GET /api/jobs/{id}`, `GET /api/countries`,
`GET /api/laws/{cc}/pdf`, `GET /api/videos/{id}`, `GET /api/status`.

## Setup and tests

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q          # no network or API key needed
```

## Gemini backends

`config.gemini_backend()` picks, in order:

1. **Vertex AI** if `GOOGLE_CLOUD_PROJECT` / `GCP_PROJECT` is set or
   `SC_GEMINI_BACKEND=vertex`. Location `GCP_VERTEX_LOCATION` /
   `GOOGLE_CLOUD_LOCATION`, default `global`. Credentials come from ADC:
   locally `GOOGLE_APPLICATION_CREDENTIALS=.secrets/gcp-sa.json` (never commit
   it), on Cloud Run the attached service account.
2. **AI Studio** if `GEMINI_API_KEY` is set.
3. Otherwise mock.

```bash
GOOGLE_APPLICATION_CREDENTIALS=.secrets/gcp-sa.json \
GOOGLE_CLOUD_PROJECT=booming-entity-510706-e9 SC_PORT=8431 ./run.sh
```

- Models: `SC_MODEL_FAST` / `SC_MODEL_SMART`, default `gemini-3.8-flash`
  (`gemini-2.5-flash` with `SC_DATA_RESIDENCY=sg`, see below). The
  ADK agents use `SC_MODEL_SMART`; `config.apply_adk_env()` sets
  `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`
  so ADK talks to the same backend.
- Video: stage A is one multimodal call over the whole video, frames and
  audio, so speech comes from the audio track. On Vertex the video is sent
  inline up to `SC_VERTEX_INLINE_MB` (default 19), larger files go through a
  temporary object in `gs://<project>-sc-uploads` (`SC_UPLOAD_BUCKET`;
  asia-southeast1, lifecycle deletes objects after 1 day; the app also deletes
  it after the call). AI Studio uses the Files API.
- Cost: every call (direct and ADK) appends its token counts to
  `jobs/usage.log`; `SC_MAX_CALLS_PER_JOB` (default 40) is a hard cap per job.

## Privacy and data handling

What the app does (details and the data flow: `docs/GOVERNANCE.md`):

- **Video**: deleted as soon as its check ends, whether it finished, failed or
  was cancelled (`app/jobs.py` `run_job` finally → `app/retention.py`
  `purge_video`): the file, its meta, anything under `uploads/<id>*`, and the
  upload-bucket object a large Vertex call used. Uploads never analysed go
  after `SC_UNUSED_VIDEO_TTL_MIN` (default 60). The report's
  `privacy.video_deleted_at` records when. The UI plays the browser's own copy
  of the file; after a reload it shows the timeline without the player.
- **Reports**: deleted `SC_REPORT_TTL_DAYS` (default 7) after creation, by a
  sweeper (startup + every 30 min) and lazily on read (404 "This report
  expired and was deleted"). `DELETE /api/jobs/{id}` ("Delete this report
  now") deletes the report and everything stored for it at once.
- **Logs**: ids, sizes, counts, durations, model names and token counts only
  (`app/logsafe.py`; `tests/test_logs.py` checks a canary phrase never reaches
  a log line or `jobs/usage.log`).
- **Provenance**: each report's `provenance` records the laws used (id,
  title, draft/reviewed, source PDF sha256, provisions file date), model ids
  and Vertex location, the app git sha and created_at (`app/provenance.py`;
  `deploy.sh` writes `data/build_info.json` since the image has no `.git`).
  Results show it under "How this report was made".
- **`/privacy`**: a plain-language page (public, no access code) whose
  configuration-dependent sentences (TTLs, Gemini location, caching) come from
  config (`app/main.py` `privacy_values`). `SC_APP_REGION` (set by
  `deploy.sh`) names the Cloud Run region; set `SC_VERTEX_CACHE_DISABLED=1`
  once the cacheConfig step below is done.
- **Upload bucket**: soft delete is off (`retentionDurationSeconds: 0`, set
  2026-10-05), so deleted objects are gone, not recoverable for 7 days; the
  lifecycle rule deletes anything left after 1 day.

### What Vertex AI does with prompts (Google's docs, checked 2026-10-05)

- **No training**: "Google won't use your data to train or fine-tune any AI/ML
  models without your prior permission or instruction" (Service Specific Terms,
  "Training Restriction").
  <https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/zero-data-retention>
- **In-memory caching**: by default Gemini models cache inputs, outputs and
  derived data in memory (not at rest), isolated per project, with a 24-hour
  TTL. It can be turned off per project with the `cacheConfig` API (same page).
- **Abuse monitoring**: if Google's safety classifiers flag a prompt, Google may
  log it for up to 90 days to check for policy violations (not used for
  training). Customers under the online Google Cloud Platform Terms of Service
  (this project) are in scope; zero data retention needs an exception from
  Google. <https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/abuse-monitoring>
- Not used here: request-response logging to BigQuery (off by default),
  Grounding with Google Search/Maps, Live API session resumption, the
  Interactions API.
- **Where**: the default location `global` "may be processed in any Google
  Cloud location around the world, and therefore don't provide any data
  residency guarantees"; regional endpoints keep ML processing in that
  region's jurisdiction.
  <https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/data-residency>,
  <https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations>
- Cloud Run request logs (URL with ids, IP address, time) stay in Cloud
  Logging's `_Default` bucket for 30 days.
  <https://docs.cloud.google.com/logging/quotas#logs_retention_periods>

**Manual step for zero data retention (not done yet).** On 2026-10-06 the
project's cache is on (`GET .../cacheConfig` returns no `disableCache`), and
the app's service account can't change it (403, missing
`aiplatform.cacheConfigs.update`). Someone with `roles/aiplatform.admin` on the
project runs:

```bash
gcloud auth login            # a user with roles/aiplatform.admin
curl -X PATCH -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/booming-entity-510706-e9/cacheConfig \
  -d '{"name": "projects/booming-entity-510706-e9/cacheConfig", "disableCache": true}'
# check: the GET returns "disableCache": true (the setting applies to all regions)
curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/booming-entity-510706-e9/cacheConfig
```

For full zero data retention, also ask Google (support or account team) for
an abuse-monitoring prompt-logging exception. Until both are done, don't claim
zero data retention.

### Data residency: `SC_DATA_RESIDENCY=sg`

By default Gemini runs on the `global` endpoint with `gemini-3.8-flash`: no
residency guarantee. `SC_DATA_RESIDENCY=sg` switches to location
`asia-southeast1` with `gemini-2.5-flash` (it overrides `GOOGLE_CLOUD_LOCATION`;
`SC_MODEL_*` still override the model). `gemini-3.8-flash` is not served in
asia-southeast1 (404; Google lists in-region ML processing for it only in the
US and EU multi-regions), while `gemini-2.5-flash` (128k context) has ML
processing in Singapore. Trade-off: everything then stays in Singapore
(Cloud Run, the upload bucket, RAG Engine already do), but the older model
is weaker at the video inventory and the legal assessment, so expect more
misses and more "check manually" moments. The default stays `global` +
`gemini-3.8-flash`.

## Consistent ratings: severity floors

The same video should get the same rating every run. The legal judgement calls
(stage B, the ADK assessor and critic, the assessor fallback) run at
temperature `SC_JUDGE_TEMPERATURE` (default 0) with a fixed seed
(`SC_MODEL_SEED`), and after the critic a fixed table in code
(`app/floors.py`) raises a finding to its floor. A floor never lowers a
rating, is skipped when an exception (news, education...) applies, and needs
a cited provision that genuinely matches: a content provision of an in-force
law (not a bill or guideline) that targets the leaf and binds the person who
publishes (creator, advertiser or both).

| Leaf | Floor | Only when the moment... |
|---|---|---|
| `gambling.promotion` | red | (always: promoting gambling is an offence under the cited law) |
| `health.cure_claim` | red | names a serious disease, or tells viewers to stop their medicine |
| `health.misinformation` | red | tells viewers to stop their medicine or treatment |

A raised finding says so (`floor_note`, "Fixed severity rule" in the UI, and a
feed event).

## Law search (Vertex AI RAG Engine)

Retrieval path B (`retrieval.search`, the assessor's `search_law` tool) queries
Vertex AI RAG Engine on the Vertex backend. Gemini File Search is not used
because it only works with an AI Studio key (google-genai rejects it on
Vertex). There is one corpus per country, `<SC_SEARCH_STORE>-<cc>` (default
`streamclearance-laws-my`, ...), in `SC_SEARCH_LOCATION` (default
asia-southeast1, a preview region). Each corpus holds one document per
provision and one per page of every law in the registry. A provision hit
maps back to its provision. A full-text hit maps to the best provision on
that law page, or comes back as a passage (law, page, snippet).

```bash
GOOGLE_APPLICATION_CREDENTIALS=.secrets/gcp-sa.json GOOGLE_CLOUD_PROJECT=booming-entity-510706-e9 \
  .venv/bin/python -m tools.index_laws          # idempotent; --country MY, --dry-run
```

Run it after changing provisions or law texts, then commit
`data/search_index.json`. It records what was uploaded (file name → hash and
metadata) and which corpus belongs to each country, and the app reads it to
map hits back. Documents are staged in the upload bucket and imported in
bulk.

`SC_LAW_SEARCH=bm25` forces the local keyword search. Mock mode, the AI Studio
backend and countries without a corpus also use BM25, and so does any
RAG Engine error (it is logged, and the country report gets a note).
The service account needs `roles/aiplatform.user` (RAG Engine) and object
write access to the upload bucket (only for indexing).

Cost: indexing embeds about 1.2 M characters with text-embedding-005, and each
query embeds a few words, so both are fractions of a cent. The corpora live in
RAG Engine's managed vector DB: Spanner, "basic" tier (100 processing units,
the project default). Google bills that Spanner instance as long as it exists.
The docs only say so explicitly for GA regions, and asia-southeast1 is a
preview region, so check the billing report. To stop the cost, delete the
corpora, or set the tier to `Unprovisioned` (`UpdateRagEngineConfig`), which
deletes all RAG data in that region. The app then falls back to BM25.

## Cloud Run

`./deploy.sh` builds the image with Cloud Build into Artifact Registry
(`asia-southeast1-docker.pkg.dev/<project>/streamclearance`) and deploys the
`streamclearance` service in asia-southeast1 with the
`hercules-deployer@` service account, Vertex env vars, 2 vCPU / 2 GiB,
`--max-instances 1`. The service is public (`--allow-unauthenticated`) but
every `/api` route needs the access code (`SC_ACCESS_CODE`); `deploy.sh`
generates it once into `.secrets/access_code.txt` (gitignored).

Uploads, jobs and `usage.log` live on the instance's local disk (in memory on
Cloud Run). With `--max-instances 1` that's fine for a demo, but they vanish
when the instance scales to zero (`--min-instances 0`) or is replaced by a
new revision.

Cloud Run caps HTTP/1 request bodies at 32 MiB, so uploads above that fail on
Cloud Run even though `SC_MAX_UPLOAD_MB` is 200 (direct-to-GCS uploads would
lift this).

## Building rule cards

```bash
.venv/bin/python -m tools.extract_text                 # already run; outputs committed
.venv/bin/python -m tools.build_rules --country IN --dry-run   # mocked Gemini, no key
.venv/bin/python -m tools.build_rules --country IN             # real run, needs key
```

Every card is written with `needs_review: true`. Cards whose quote can't be
found on the cited page are dropped and logged. A person still has to check each
card against the PDF before it is used. `page` is the 1-based PDF page, not the
page number printed in the gazette.
