# StreamClearance

Upload a short video, pick APAC countries, get a per-country risk report with
timestamped flags tied to the exact law section and page. Advisory only, not
legal advice. Most of these laws put duties on platforms, so the practical risk
is takedown or geo-blocking in that country.

## Status: M0–M5 working locally (mock mode until a Gemini key is added)

| Path | What it does |
|---|---|
| `app/config.py` | Paths, model IDs (`SC_MODEL_FAST`, `SC_MODEL_SMART`), country → law map |
| `app/schemas.py` | Pydantic models: `RuleCard`, `Segment`, `Inventory`, `Finding`, `CountryReport`, `Report`, `Category` |
| `app/gemini.py` | The only module that calls Gemini (google-genai, AI Studio key) |
| `app/openrouter_backend.py` | **Temporary** OpenRouter bridge (free models only); see below |
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

No cloud deployment yet (no GCS, Vertex or Cloud Run).

## Running the app

```bash
./run.sh                      # http://localhost:8430  (SC_PORT to change)
```

Without `GEMINI_API_KEY` (or with `SC_MOCK=1`) everything runs in **mock mode**:
stage A returns a canned inventory scaled to the real video length, stage B
returns canned findings (plus one invented card ID the validator must drop),
and the drift check returns one canned "increased" example. The UI shows a
MOCK MODE banner. Set `GEMINI_API_KEY` in `.env` and restart to use Gemini
(video goes through the AI Studio Files API).

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

## Adding the Gemini key

```bash
cp .env.example .env         # .env is gitignored
# put your AI Studio key in GEMINI_API_KEY=
```

## Temporary OpenRouter bridge

Until a Gemini AI Studio key exists, `app/gemini.py` can dispatch to
`app/openrouter_backend.py`. Provider order: `SC_MOCK=1` forces mock; else
`SC_PROVIDER` (`gemini|openrouter|mock`) if set; else Gemini if
`GEMINI_API_KEY` is set; else OpenRouter if `OPENROUTER_API_KEY` is set; else mock.

- **Free models only.** `SC_OR_MODEL` (default `google/gemma-4-31b-it:free`,
  fallback `google/gemma-4-26b-a4b-it:free` on 429/5xx/unavailable). Any model id
  not ending in `:free` is refused before a request is sent. Every call asks for
  usage accounting and is logged to `jobs/usage.log`; a non-zero cost stops all
  further OpenRouter calls with an error.
- Gemma 4 takes no audio, so speech is transcribed first by
  `SC_OR_AUDIO_MODEL` (default `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`)
  and merged into the stage A prompt. If that fails the inventory gets
  `speech_transcribed: false` and the report shows a note.
- Video is transcoded to ≤480p / mono 32k audio (ffmpeg) and sent inline (max
  20 MB). If OpenRouter rejects video, frames every 2 s (max 60) are sent instead.
- JSON: `json_schema` → `json_object` → schema in the prompt, then one repair
  call, validated with Pydantic.
- Free-tier limits: concurrency defaults to 1, at least 3.5 s between calls,
  `Retry-After` honoured.

### Removing OpenRouter

1. `git rm app/openrouter_backend.py tests/test_openrouter.py`
2. Delete every line containing `OPENROUTER-TEMP`:
   `grep -rn OPENROUTER-TEMP app` (app/config.py ×2, app/gemini.py ×3,
   app/schemas.py, app/pipeline.py, app/static/app.js).
3. Remove `"openrouter"` from `PROVIDERS` in `app/config.py`, the OpenRouter
   block from `.env.example`, this section, and the
   table row above; drop `OPENROUTER_API_KEY` from `.env`.
4. `.venv/bin/pytest -q`.

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
