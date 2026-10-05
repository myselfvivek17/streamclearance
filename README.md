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

- Models: `SC_MODEL_FAST` / `SC_MODEL_SMART`, default `gemini-3.8-flash`. The
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
