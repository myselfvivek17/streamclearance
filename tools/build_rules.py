"""Draft rule cards for one country from its extracted law text.

    python -m tools.build_rules --country IN [--dry-run] [--out-dir DIR]

Pipeline: data/laws/text/<source>.json -> page-preserving chunks -> Gemini
(JSON-schema output) -> post-validation -> data/rules/<cc>.json.

Post-validation drops (and logs) any card whose page does not exist or whose
`quote` does not appear on that page. Comparison ignores case, all whitespace
and typographic quote/dash variants, because pypdf inserts stray spaces inside
words ("com pliance", "sub -section") and letter-spaces some PDFs (Pakistan).

--dry-run swaps Gemini for a deterministic mock built from the real text, so
the whole pipeline runs without an API key. Unless --out-dir is given, dry-run
output goes to data/rules/<cc>.dryrun.json so real cards are never clobbered.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app import config, gemini
from app.schemas import AppliesTo, Category, RuleCard

log = logging.getLogger("build_rules")

MAX_PAGES_PER_CHUNK = 8
MAX_CHARS_PER_CHUNK = 30_000
QUOTE_MAX = 300


class RuleCardDraft(BaseModel):
    """What Gemini returns; id/country/law/needs_review are filled in by us."""
    section: str
    page: int
    category: Category
    conduct: str
    applies_to: AppliesTo
    platform_duty: str
    penalty: str
    quote: str


# ---------------------------------------------------------------- text utils

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "‒": "-", "‑": "-", "­": "",
    " ": " ",
})
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, unify quotes/dashes, and drop all whitespace."""
    return _WS.sub("", text.translate(_TRANSLATE).lower())


def tidy(text: str) -> str:
    """Collapse whitespace for prompts (keeps words readable, saves tokens)."""
    return _WS.sub(" ", text).strip()


def quote_on_page(quote: str, page_text: str) -> bool:
    q = normalise(quote)
    return bool(q) and q in normalise(page_text)


def load_pages(country: config.CountryLaw) -> dict[int, str]:
    pages = json.loads(country.text_path.read_text(encoding="utf-8"))
    return {p["page"]: p["text"] for p in pages}


def chunk_pages(
    pages: dict[int, str],
    max_pages: int = MAX_PAGES_PER_CHUNK,
    max_chars: int = MAX_CHARS_PER_CHUNK,
) -> list[list[int]]:
    chunks: list[list[int]] = []
    cur: list[int] = []
    size = 0
    for n in sorted(pages):
        length = len(tidy(pages[n]))
        if cur and (len(cur) >= max_pages or size + length > max_chars):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(n)
        size += length
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- prompting

SYSTEM = """You turn statute text into structured "rule cards" for a tool that warns \
online video creators which content may be taken down or geo-blocked in a country.

Frame risk honestly:
- Most online-safety laws place duties on PLATFORMS (intermediaries, service providers), \
not on creators. For such provisions, applies_to = "platform", `conduct` describes the \
category of content targeted, and `platform_duty` states exactly what the platform must \
do (e.g. "remove or disable access within 36 hours of a court or government order", \
"remove within 24 hours of a complaint"), using the timeframes in the text.
- Only use applies_to = "creator" or "both" when the text itself makes the person who \
posts or creates content liable (an offence, fine or order directed at the user). Put \
that liability in `penalty`; if the penalty only falls on the platform, say so there.
- Never invent duties, timeframes or penalties. If the page does not state one, write \
"not specified on this page".

Selection:
- Only provisions relevant to ONLINE VIDEO CONTENT: what content is prohibited or \
harmful, takedown/blocking duties, and offences for posting content.
- Skip definitions, procedure, committees, fees and administration unless they define \
a harmful-content category used elsewhere.
- Ignore footnotes that reproduce superseded earlier wording.
- One card per distinct provision + category. Fewer, accurate cards beat many vague ones.

Citations:
- `page` is the number in the "=== PAGE n ===" marker the quote comes from.
- `section` is the section/rule/clause reference as written (e.g. "Rule 3(1)(b)(ii)").
- `quote` must be copied VERBATIM from that single page, at most 300 characters, no \
ellipses, no paraphrase. It will be machine-checked against the page text."""


def build_prompt(country: config.CountryLaw, pages: dict[int, str], chunk: list[int]) -> str:
    body = "\n\n".join(f"=== PAGE {n} ===\n{tidy(pages[n])}" for n in chunk)
    caveat = f"\nCaveat: {country.label}" if country.label else ""
    return (
        f"Country: {country.name} ({country.code})\nLaw: {country.law}{caveat}\n"
        f"Pages {chunk[0]}-{chunk[-1]} follow. Return a JSON array of rule cards "
        f"(empty array if none of these pages are relevant).\n\n{body}"
    )


Generate = Callable[[str], list[dict]]


def gemini_generate(prompt: str) -> list[dict]:
    return gemini.generate_json(
        prompt, list[RuleCardDraft], model=config.SC_MODEL_SMART, system_instruction=SYSTEM
    )


def mock_generate_factory(pages: dict[int, str]) -> Generate:
    """Deterministic stand-in for Gemini: one real quote and one invented one per chunk."""
    marker = re.compile(r"=== PAGE (\d+) ===")

    def _generate(prompt: str) -> list[dict]:
        nums = [int(m) for m in marker.findall(prompt)]
        page = next((n for n in nums if len(tidy(pages[n])) > 200), nums[0])
        words = tidy(pages[page]).split(" ")
        mid = len(words) // 3
        quote = " ".join(words[mid: mid + 20])[:QUOTE_MAX]
        common = dict(
            section="(mock)", applies_to="platform",
            platform_duty="(mock) remove on notice", penalty="(mock) not specified",
        )
        return [
            dict(common, page=page, category="other",
                 conduct="(mock) real quote from page", quote=quote),
            dict(common, page=page, category="hate_or_incitement",
                 conduct="(mock) invented quote, must be rejected",
                 quote="Creators shall be imprisoned for any video that offends anyone."),
        ]

    return _generate


# ---------------------------------------------------------------- pipeline

def build_cards(country: config.CountryLaw, generate: Generate) -> tuple[list[RuleCard], list[dict]]:
    pages = load_pages(country)
    cards: list[RuleCard] = []
    rejected: list[dict] = []
    seen: set[tuple[int, str]] = set()

    for chunk in chunk_pages(pages):
        log.info("%s: pages %d-%d", country.code, chunk[0], chunk[-1])
        for raw in generate(build_prompt(country, pages, chunk)):
            reason = None
            try:
                draft = RuleCardDraft.model_validate(raw)
            except ValidationError as e:
                draft, reason = None, f"schema: {e.errors()[0]['msg']}"
            if draft is not None:
                if draft.page not in pages:
                    reason = f"page {draft.page} does not exist"
                elif len(draft.quote) > QUOTE_MAX:
                    reason = f"quote longer than {QUOTE_MAX} chars"
                elif not quote_on_page(draft.quote, pages[draft.page]):
                    reason = f"quote not found on page {draft.page}"
                elif (draft.page, normalise(draft.quote)) in seen:
                    reason = "duplicate"
            if reason:
                log.warning("REJECT %s: %s | %r", country.code, reason, raw)
                rejected.append({"reason": reason, "card": raw})
                continue
            seen.add((draft.page, normalise(draft.quote)))
            cards.append(RuleCard(
                id=f"{country.code}-{len(cards) + 1:03d}",
                country=country.code,
                law=country.law,
                needs_review=True,
                **draft.model_dump(),
            ))
    return cards, rejected


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--country", required=True, choices=sorted(config.COUNTRIES))
    ap.add_argument("--dry-run", action="store_true", help="mock Gemini; no API key needed")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    country = config.COUNTRIES[args.country]
    if not country.text_path.exists():
        log.error("missing %s; run `python -m tools.extract_text` first", country.text_path)
        return 1
    generate = mock_generate_factory(load_pages(country)) if args.dry_run else gemini_generate

    try:
        cards, rejected = build_cards(country, generate)
    except gemini.GeminiUnavailable as e:
        log.error("%s (or use --dry-run)", e)
        return 2

    out_dir = args.out_dir or config.RULES_DIR
    name = f"{args.country.lower()}.dryrun.json" if args.dry_run and not args.out_dir \
        else f"{args.country.lower()}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    out.write_text(
        json.dumps([c.model_dump(mode="json") for c in cards], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{args.country}: {len(cards)} cards kept, {len(rejected)} rejected -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
