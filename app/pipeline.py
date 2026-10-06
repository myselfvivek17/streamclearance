"""Analysis pipeline.

Stage A: video -> Inventory (one Gemini video call per video).
Stage B: Inventory + one country's rule cards -> CountryReport (one call per
country, run concurrently under a small cap). Findings citing card IDs that
are not in that country's cards are dropped by `validate_findings`.
Translation drift: each flagged line with speech is translated into the
country's main language, back-translated blind, and Gemini judges whether the
risk changed.

Every stage has a deterministic mock used when `config.mock_mode()` is true
(no Gemini backend configured, or SC_MOCK=1), so the whole app runs without a key.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app import config, gemini, metrics, provisions, taxonomy
from app.schemas import (
    RISK_ORDER, Category, CountryReport, Drift, DriftCheck, Finding, Inventory, Risk, RuleCard,
    RulesStatus, Segment, Uncovered,
)
from app.validate import validate_findings

log = logging.getLogger(__name__)

MOCK_UNKNOWN_DURATION_S = 60.0


# ================================================================ stage A

class SegmentDraft(BaseModel):
    """One segment as the model returns it; legacy categories are derived from leaves."""
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    transcript: str = ""
    language: str = ""
    claims: list[str] = []
    visuals: str = ""
    on_screen_text: str = ""
    leaves: list[str] = []
    idioms: list[str] = []


class StageADraft(BaseModel):
    """What Gemini returns for stage A; duration comes from ffprobe, not the model."""
    languages: list[str] = []
    segments: list[SegmentDraft] = []


STAGE_A_SYSTEM = f"""You build a factual, timestamped inventory of a short video for a \
content-compliance tool. Do not judge legality; only describe what is there.

Split the video into consecutive segments of roughly 3-15 seconds at natural breaks \
(sentence or scene changes). For each segment give:
- start_s / end_s in seconds from the start of the video (no gaps, no overlaps)
- transcript: verbatim speech in the original language ("" if none)
- language: ISO 639-1 code of the speech ("" if none)
- claims: factual or health/financial claims made, one per item
- visuals: what is shown (people, actions, symbols, products, logos, flags, weapons, etc.)
- on_screen_text: captions, overlays or signs, verbatim
- leaves: zero or more ids from the content taxonomy below that the segment might \
plausibly touch, most specific first (e.g. a casino promo code is "gambling.promotion", \
not "drugs.promotion"; "this drink cures diabetes" is "health.cure_claim"). Use the ids \
exactly as written. Leave empty for clearly neutral segments.
- idioms: figures of speech or slang in the speech that could read as literal violence, \
crime, insult or threat if translated word for word (e.g. "this offer is a bomb", \
"it's a killer deal"), quoted verbatim; [] if none.

Content taxonomy:
{taxonomy.prompt_list()}"""


def stage_a(video_path: Path, duration_s: float | None = None, *, mock: bool | None = None) -> Inventory:
    """Video -> Inventory. duration_s comes from ffprobe and may be unknown (None)."""
    mock = config.mock_mode() if mock is None else mock
    if mock:
        return mock_inventory(duration_s)

    mime = config.VIDEO_TYPES.get(video_path.suffix.lower(), "video/mp4")
    dur_hint = f"The video is {duration_s:.1f} seconds long. " if duration_s else ""
    raw = gemini.generate_json_with_video(
        video_path, mime,
        f"{dur_hint}Return the inventory as JSON.",
        StageADraft,
        model=config.SC_MODEL_FAST,
        system_instruction=STAGE_A_SYSTEM,
    )
    draft = StageADraft.model_validate(raw)
    segments = []
    for d in sorted(draft.segments, key=lambda s: s.start_s):
        unknown = [x for x in d.leaves if not taxonomy.is_leaf(str(x).strip().lower())]
        if unknown:
            log.warning("stage A: ignoring %d unknown taxonomy ids", len(unknown))
        try:
            segments.append(Segment.model_validate(d.model_dump()))
        except ValidationError as e:
            log.warning("stage A: dropping malformed segment: %s", e.errors()[0]["type"])
    if duration_s is None:
        duration_s = max((s.end_s for s in segments), default=0.0)
    kept = []
    for s in segments:
        if s.start_s >= duration_s:
            log.warning("stage A: dropping segment at %.1fs past duration %.1fs", s.start_s, duration_s)
            continue
        kept.append(s.model_copy(update={"end_s": min(s.end_s, duration_s)}))
    return Inventory(
        duration_s=duration_s, languages=draft.languages, segments=kept,
        speech_transcribed=True,  # Gemini hears the audio track in the same call
    )


_MOCK_SEGMENTS = [
    dict(transcript="Hey everyone, welcome back! Today I'm unboxing the new festival snack box.",
         visuals="Creator at a kitchen table holding a colourful snack box.",
         on_screen_text="FESTIVAL SNACK BOX UNBOXING", claims=[], leaves=[]),
    dict(transcript="Honestly, fasting this month is pointless. Just eat the snacks, God won't mind.",
         visuals="Creator laughing, waving a pastry in front of a religious festival poster.",
         on_screen_text="", claims=["Religious fasting is pointless"],
         leaves=["religion.insult"]),
    dict(transcript="And these snacks cure diabetes. Doctors don't want you to know this.",
         visuals="Close-up of a snack label; creator points at it.",
         on_screen_text="CURES DIABETES?!", claims=["The snacks cure diabetes"],
         leaves=["health.cure_claim"]),
    dict(transcript="Use my code at LuckySpin Casino for a hundred free bets tonight.",
         visuals="Phone screen showing an online casino slot game.",
         on_screen_text="LUCKYSPIN.BET  CODE: SNACK100", claims=["100 free bets with code"],
         leaves=["gambling.promotion"]),
    dict(transcript="This deal is a killer, it's basically a crime not to grab it.",
         visuals="Creator holding the box up to the camera, smiling.",
         on_screen_text="LINK IN BIO", claims=[], leaves=[],
         idioms=["This deal is a killer", "basically a crime not to grab it"]),
    dict(transcript="Like and subscribe, see you next week!",
         visuals="End card with channel logo.", on_screen_text="SUBSCRIBE", claims=[], leaves=[]),
]


def mock_inventory(duration_s: float | None) -> Inventory:
    """Deterministic plausible inventory scaled to the real duration."""
    duration = duration_s if duration_s and duration_s > 0 else MOCK_UNKNOWN_DURATION_S
    n = max(3, min(8, math.ceil(duration / 10)))
    step = duration / n
    segments = []
    for i in range(n):
        tpl = _MOCK_SEGMENTS[i % len(_MOCK_SEGMENTS)]
        segments.append(Segment(
            start_s=round(i * step, 2),
            end_s=round(duration if i == n - 1 else (i + 1) * step, 2),
            language="en",
            **tpl,
        ))
    return Inventory(duration_s=round(duration, 2), languages=["en"], segments=segments)



# ================================================================ rule cards

_dryrun_lock = threading.Lock()


def load_cards(code: str) -> tuple[list[RuleCard], RulesStatus]:
    """Cards derived from provisions if the country has any (data/provisions),
    else a rule-card file if present, else dry-run cards (generated on demand).

    Drafted provisions/cards carry needs_review=true; a country is only
    "reviewed" once every card has been signed off.
    """
    country = config.COUNTRIES[code]
    if provisions.has_provisions(code):
        cards = provisions.cards_from_provisions(code)
        status = "draft/unreviewed" if any(c.needs_review for c in cards) else "reviewed"
        return cards, status
    if country.rules_path.exists():
        cards = _read_cards(country.rules_path)
        status: RulesStatus = "draft/unreviewed" if any(c.needs_review for c in cards) else "reviewed"
        return cards, status

    with _dryrun_lock:
        if not country.dryrun_rules_path.exists():
            from tools import build_rules

            log.warning("%s: no rule cards; generating dry-run cards", code)
            # The dry-run mock deliberately emits invented quotes; their REJECT lines are noise here.
            br_log = logging.getLogger("build_rules")
            level, br_log.level = br_log.level, logging.ERROR
            try:
                cards, rejected = build_rules.build_cards(
                    country, build_rules.mock_generate_factory(build_rules.load_pages(country)))
            finally:
                br_log.setLevel(level)
            log.info("%s: %d dry-run cards (%d rejected)", code, len(cards), len(rejected))
            country.dryrun_rules_path.parent.mkdir(parents=True, exist_ok=True)
            country.dryrun_rules_path.write_text(json.dumps(
                [c.model_dump(mode="json") for c in cards], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
    return _read_cards(country.dryrun_rules_path), "draft/unreviewed"


def _read_cards(path: Path) -> list[RuleCard]:
    return [RuleCard.model_validate(c) for c in json.loads(path.read_text(encoding="utf-8"))]


# ================================================================ stage B

STAGE_B_SYSTEM = """You check a video's timestamped inventory against ONE country's \
rule cards and report which moments risk being taken down or geo-blocked there.

Rules:
- Cite ONLY rule card ids from the list you are given (field "id"). Never invent ids, \
sections or laws.
- A card fits only if its `conduct` text actually describes this kind of content. Sharing \
a broad category is NOT enough (e.g. a card about dangerous drugs does not cover a casino \
promotion; a card about hostility between groups does not cover a false health claim). \
If no card fits a moment, do NOT flag it: the report lists it separately as "not covered \
by this law". Never stretch a card to make something fit.
- The finding's `category` must equal the `category` of at least one cited card. Cards \
with category "other" (general procedures) may be cited in addition, never alone.
- A finding covers one moment (start_s/end_s inside the video, taken from the \
inventory) and one category; rule_card_ids lists every card it relies on.
- risk: "red" = content clearly falls in a category the cited card targets (likely \
takedown/geo-block); "amber" = arguable or context-dependent; do not emit "green" findings.
- Frame risk honestly: most cards place duties on PLATFORMS (remove/disable access). Say \
what the platform would have to do. Only mention consequences for the creator when a \
cited card's applies_to is "creator" or "both", and then quote its penalty.
- rationale: 1-3 sentences tying the specific words/visuals to the card's conduct.
- suggested_fix: a concrete edit (cut, reword, blur, add context/disclaimer) that would \
lower the risk while keeping the creator's intent.
- Satire, news and educational context can lower risk; say so in the rationale.
- Return an empty array if nothing is risky."""


def worst_risk(findings: list[Finding]) -> Risk:
    return max((f.risk for f in findings), key=RISK_ORDER.__getitem__, default="green")


def stage_b(
    inventory: Inventory,
    country: str,
    cards: list[RuleCard],
    *,
    rules_status: RulesStatus = "reviewed",
    mock: bool | None = None,
) -> CountryReport:
    """Inventory + one country's cards -> validated CountryReport."""
    mock = config.mock_mode() if mock is None else mock
    law = config.COUNTRIES[country]
    raw = _mock_findings(inventory, country, cards) if mock else _gemini_findings(inventory, law, cards)

    parsed: list[Finding] = []
    for item in raw:
        try:
            parsed.append(Finding.model_validate({**item, "creator_liability": False}))
        except ValidationError as e:
            log.warning("%s: dropping malformed finding: %s", country, e.errors()[0]["type"])
    findings = validate_findings(parsed, cards, inventory.duration_s)
    findings = [f for f in findings if f.risk != "green"]

    by_id = {c.id: c for c in cards}
    findings = [
        f.model_copy(update={"creator_liability": any(
            by_id[cid].applies_to in ("creator", "both") for cid in f.rule_card_ids)})
        for f in sorted(findings, key=lambda f: (f.start_s, -RISK_ORDER[f.risk]))
    ]
    cited = sorted({cid for f in findings for cid in f.rule_card_ids})
    uncovered = _uncovered(inventory, findings)
    return CountryReport(
        country=country,
        law=law.law,
        label=law.label,
        overall_risk=worst_risk(findings),
        findings=findings,
        rules_status=rules_status,
        risk_framing=_risk_framing(findings),
        cited_cards=[by_id[cid] for cid in cited],
        dropped_findings=len(raw) - len(findings),
        uncovered=uncovered,
    )


def _uncovered(inventory: Inventory, findings: list[Finding]) -> list[Uncovered]:
    """Moments stage A tagged as sensitive that no finding covers in this country.

    Shown as "not covered by this law" so the report never implies a moment is
    safe just because this country's statute has no matching provision.
    """
    out: list[Uncovered] = []
    for seg in inventory.segments:
        for cat in seg.categories:
            if cat == Category.other:
                continue
            hit = any(f.category == cat and f.start_s < seg.end_s and seg.start_s < f.end_s
                      for f in findings)
            if not hit:
                out.append(Uncovered(start_s=seg.start_s, end_s=seg.end_s, category=cat))
    return out


def _risk_framing(findings: list[Finding]) -> str:
    if not findings:
        return "No moments matched this country's rule cards."
    if any(f.creator_liability for f in findings):
        return ("Takedown or geo-blocking by the platform is the main risk. At least one cited "
                "provision also applies to the person who posts the content; see its penalty.")
    return ("The cited provisions place duties on platforms, so the practical risk is takedown "
            "or geo-blocking of this video in this country, not liability for the creator.")


def _gemini_findings(inventory: Inventory, law: config.CountryLaw, cards: list[RuleCard]) -> list[dict]:
    card_json = json.dumps(
        [c.model_dump(mode="json", exclude={"needs_review", "country", "law"}) for c in cards],
        ensure_ascii=False)
    caveat = f" ({law.label})" if law.label else ""
    prompt = (
        f"Country: {law.name} ({law.code})\nLaw: {law.law}{caveat}\n"
        f"Video duration: {inventory.duration_s:.1f}s\n\n"
        f"RULE CARDS:\n{card_json}\n\n"
        f"VIDEO INVENTORY:\n{inventory.model_dump_json(exclude={'duration_s'})}\n\n"
        "Return the findings as a JSON array."
    )
    out = gemini.generate_json(
        prompt, list[Finding], model=config.SC_MODEL_SMART, system_instruction=STAGE_B_SYSTEM)
    return [x for x in out if isinstance(x, dict)] if isinstance(out, list) else []


_MOCK_RISK: dict[Category, dict[str, Risk]] = {
    Category.religious_offence: {"IN": "amber", "AU": "green", "MY": "red", "SG": "amber", "PK": "red"},
    Category.misinformation_public_order: {"IN": "amber", "AU": "amber", "MY": "amber", "SG": "red", "PK": "amber"},
    Category.illegal_goods_or_gambling: {"IN": "amber", "AU": "amber", "MY": "red", "SG": "red", "PK": "red"},
}
_MOCK_FIX: dict[Category, str] = {
    Category.religious_offence: "Cut the line about fasting, or reframe it as your own experience without mocking the practice.",
    Category.misinformation_public_order: "Remove the cure claim or replace it with a sourced, accurate statement.",
    Category.illegal_goods_or_gambling: "Drop the casino promo code and on-screen link for this market, or geo-restrict the upload.",
}


def _mock_findings(inventory: Inventory, country: str, cards: list[RuleCard]) -> list[dict]:
    """Deterministic findings, plus one invented card id that the validator must drop."""
    if not cards:
        return []
    out: list[dict] = []
    for seg in inventory.segments:
        for cat in seg.categories:
            risk = _MOCK_RISK.get(cat, {}).get(country, "amber")
            if risk == "green":
                continue
            card = next((c for c in cards if c.category == cat), None)
            if card is None:
                continue
            out.append(dict(
                start_s=seg.start_s, end_s=seg.end_s, category=cat.value,
                rule_card_ids=[card.id], risk=risk,
                suggested_fix=_MOCK_FIX.get(cat, "Review this moment before publishing."),
                rationale=(f"(mock) \"{seg.transcript[:80]}\" may fall under {card.section} "
                           f"({card.category.value.replace('_', ' ')})."),
            ))
    if inventory.segments:
        seg = inventory.segments[0]
        out.append(dict(
            start_s=seg.start_s, end_s=seg.end_s, category="other",
            rule_card_ids=[f"{country}-999"], risk="red", suggested_fix="-",
            rationale="(mock) hallucinated citation; the validator must drop this.",
        ))
    return out


# ================================================================ translation drift

class _Text(BaseModel):
    index: int
    text: str


class _Judgement(BaseModel):
    index: int
    drift: Drift
    explanation: str


TRANSLATE_SYSTEM = """You are a professional subtitle translator. Translate each item \
faithfully and naturally, the way a dubbing studio would, keeping tone, idioms and \
jokes as a native speaker would render them. Do not soften or sharpen anything. \
Return one output per input index."""

JUDGE_SYSTEM = """You review dubbed lines for a content-compliance tool. For each item \
you get the original line, its translation and an independent back-translation, plus \
the risk categories already flagged for that moment and any idioms in the line \
(figures of speech that may have been rendered literally).

drift = "increased" if the translated version is likely to read as MORE offensive, \
inflammatory, defamatory or otherwise riskier to that country's audience than the \
original (e.g. a joke that becomes an insult, an idiom that becomes literal violence); \
"reduced" if clearly less risky; otherwise "none". explanation: one or two sentences \
naming the exact words that shifted."""


def _overlaps(f: Finding, s: Segment) -> bool:
    return (f.start_s < s.end_s and s.start_s < f.end_s) or f.start_s == s.start_s


def flagged_segments(inventory: Inventory, report: CountryReport) -> list[Segment]:
    """Segments with speech that overlap a finding or carry an idiom that could turn
    literal in translation, capped (flagged ones first)."""
    speech = [s for s in inventory.segments if s.transcript.strip()]
    flagged = [s for s in speech if any(_overlaps(f, s) for f in report.findings)]
    idioms = [s for s in speech if s.idioms and s not in flagged]
    return (flagged + idioms)[: config.MAX_DRIFT_SEGMENTS]


def _categories_at(report: CountryReport, seg: Segment) -> list[str]:
    return sorted({f.category.value for f in report.findings if _overlaps(f, seg)})


def translation_check(
    inventory: Inventory, report: CountryReport, language: str, *, mock: bool | None = None,
    mock_increase: bool = False,
) -> list[DriftCheck]:
    """Translate -> blind back-translate -> judge, for one country and language."""
    mock = config.mock_mode() if mock is None else mock
    segs = flagged_segments(inventory, report)
    if not segs:
        return []
    if mock:
        return _mock_drift(segs, language, increase_first=mock_increase)

    country = config.COUNTRIES[report.country]
    items = [{"index": i, "text": s.transcript} for i, s in enumerate(segs)]
    translated = _by_index(gemini.generate_json(
        f"Translate into {language} for viewers in {country.name}.\n{json.dumps(items, ensure_ascii=False)}",
        list[_Text], model=config.SC_MODEL_FAST, system_instruction=TRANSLATE_SYSTEM), _Text)
    # Back-translate without the original in context so the model can't just copy it.
    back_items = [{"index": i, "text": translated[i].text} for i in sorted(translated)]
    back = _by_index(gemini.generate_json(
        f"Translate into English.\n{json.dumps(back_items, ensure_ascii=False)}",
        list[_Text], model=config.SC_MODEL_FAST, system_instruction=TRANSLATE_SYSTEM), _Text)
    judge_items = [
        {"index": i, "original": segs[i].transcript, "translation": translated[i].text,
         "back_translation": back[i].text, "flagged_categories": _categories_at(report, segs[i]),
         "idioms": segs[i].idioms}
        for i in sorted(translated) if i in back
    ]
    if not judge_items:
        return []
    judged = _by_index(gemini.generate_json(
        f"Audience: {country.name}, language: {language}.\n"
        f"{json.dumps(judge_items, ensure_ascii=False)}",
        list[_Judgement], model=config.SC_MODEL_SMART, system_instruction=JUDGE_SYSTEM), _Judgement)
    return [
        DriftCheck(start_s=segs[i].start_s, end_s=segs[i].end_s, language=language,
                   original=segs[i].transcript, translated=translated[i].text,
                   back_translated=back[i].text, drift=judged[i].drift,
                   explanation=judged[i].explanation)
        for i in sorted(judged) if i in translated and i in back and 0 <= i < len(segs)
    ]


def _by_index(raw, model: type[BaseModel]) -> dict[int, BaseModel]:
    out = {}
    for item in raw if isinstance(raw, list) else []:
        try:
            m = model.model_validate(item)
        except ValidationError:
            continue
        out[m.index] = m
    return out


_MOCK_BACK = {
    "Honestly, fasting this month is pointless. Just eat the snacks, God won't mind.":
        "Honestly, this month's fast is useless. Just eat the snacks, God does not care about it.",
}


def _mock_drift(segs: list[Segment], language: str, *, increase_first: bool) -> list[DriftCheck]:
    out = []
    for i, s in enumerate(segs):
        increased = increase_first and i == 0
        out.append(DriftCheck(
            start_s=s.start_s, end_s=s.end_s, language=language, original=s.transcript,
            translated=f"(mock {language} translation) {s.transcript}",
            back_translated=_MOCK_BACK.get(s.transcript, s.transcript) if increased else s.transcript,
            drift="increased" if increased else "none",
            explanation=(
                "(mock) The dubbed line turns a throwaway joke into a flat statement that the "
                "religious practice is useless and that God doesn't care, which reads as more "
                "dismissive to observant viewers." if increased else
                "(mock) Meaning and tone survive translation."),
        ))
    return out


async def run_translation(
    inventory: Inventory, reports: dict[str, CountryReport], *, mock: bool | None = None,
) -> list[str]:
    """Fill report.translation for every country with a drift language. Returns errors."""
    mock = config.mock_mode() if mock is None else mock
    sem = asyncio.Semaphore(max(1, config.GEMINI_CONCURRENCY))
    tasks = [(cc, lang) for cc, rep in reports.items()
             for lang in config.COUNTRIES[cc].drift_languages if rep.findings]
    # Mock: exactly one deterministic "increased" example, on the first eligible pair.
    first = next((t for t in tasks if flagged_segments(inventory, reports[t[0]])), None)

    async def one(cc: str, lang: str):
        async with sem:
            with metrics.stage(f"translation:{cc}/{lang}"):
                return await asyncio.to_thread(
                    translation_check, inventory, reports[cc], lang, mock=mock,
                    mock_increase=(cc, lang) == first)

    results = await asyncio.gather(*(one(cc, lang) for cc, lang in tasks), return_exceptions=True)
    if mock and tasks:
        await asyncio.sleep(config.mock_delay_s())
    errors = []
    for (cc, lang), res in zip(tasks, results):
        if isinstance(res, BaseException):
            log.error("translation check failed for %s/%s: %s", cc, lang, type(res).__name__)
            errors.append(f"translation {cc}/{lang}: {public_error(res)}")
            continue
        reports[cc].translation.extend(res)
    return errors


# ================================================================ orchestration

async def run_countries(
    inventory: Inventory,
    countries: list[str],
    *,
    mock: bool | None = None,
    on_start: Callable[[str], Awaitable[None]] | None = None,
    on_done: Callable[[str, CountryReport | None, str | None], Awaitable[None]] | None = None,
) -> dict[str, CountryReport | Exception]:
    """Stage B for every country concurrently, at most GEMINI_CONCURRENCY at a time."""
    mock = config.mock_mode() if mock is None else mock
    sem = asyncio.Semaphore(max(1, config.GEMINI_CONCURRENCY))

    async def one(cc: str) -> CountryReport | Exception:
        async with sem:
            if on_start:
                await on_start(cc)
            try:
                cards, status = await asyncio.to_thread(load_cards, cc)
                if mock:
                    await asyncio.sleep(config.mock_delay_s())
                with metrics.stage(f"assessor:{cc}"):
                    report = await asyncio.to_thread(
                        stage_b, inventory, cc, cards, rules_status=status, mock=mock)
            except Exception as e:  # noqa: BLE001 - reported per country
                log.error("stage B failed for %s: %s", cc, type(e).__name__)
                if on_done:
                    await on_done(cc, None, public_error(e))
                return e
            if on_done:
                await on_done(cc, report, None)
            return report

    results = await asyncio.gather(*(one(cc) for cc in countries))
    return dict(zip(countries, results))


def public_error(e: Exception) -> str:
    """Error text safe to show users (never includes request details or keys)."""
    if isinstance(e, gemini.GeminiUnavailable):
        return str(e)
    if isinstance(e, ValidationError):  # its text echoes the invalid input (model output)
        return f"ValidationError: {e.error_count()} invalid field(s)"
    return f"{type(e).__name__}: {str(e)[:200]}"
