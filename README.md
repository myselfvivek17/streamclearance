# StreamClearance

Upload a short video, pick APAC countries, get a per-country risk report with
timestamped flags tied to the exact law section and page. Advisory only, not
legal advice. Most of these laws put duties on platforms, so the practical risk
is takedown or geo-blocking in that country.

## Status: M0 skeleton + M1 law ingestion (local only)

| Path | What it does |
|---|---|
| `app/config.py` | Paths, model IDs (`SC_MODEL_FAST`, `SC_MODEL_SMART`), country → law map |
| `app/schemas.py` | Pydantic models: `RuleCard`, `Segment`, `Inventory`, `Finding`, `CountryReport`, `Report`, `Category` |
| `app/gemini.py` | The only module that calls Gemini (google-genai, AI Studio key) |
| `app/validate.py` | `validate_findings()` drops findings with unknown card IDs or out-of-range timestamps |
| `tools/extract_text.py` | `data/laws/raw/*.pdf` → `data/laws/text/*.json` (`[{page, text}]`) |
| `tools/build_rules.py` | Law text → Gemini-drafted rule cards → quote/page check → `data/rules/<cc>.json` |

Countries: IN (IT Rules 2021, as updated Feb 2026), AU (Online Safety Act 2021),
MY (Online Safety Act 2025), SG (Online Safety (Relief and Accountability) Bill
2025, **the bill as introduced, so check it against the enacted Act**), PK (PECA Amendment 2025).

No web app, no video pipeline and no cloud deployment yet.

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
