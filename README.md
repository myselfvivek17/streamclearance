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
