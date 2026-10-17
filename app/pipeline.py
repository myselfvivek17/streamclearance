"""Analysis pipeline.

Stage A: video -> Inventory (one Gemini video call per video).
Stage B: Inventory + one country's rule cards -> CountryReport (one call per
country, run concurrently under a small cap). Findings citing card IDs that
are not in that country's cards are dropped by `validate_findings`.
Translation drift: for every country with a drift language and a video with
speech (findings or not), up to MAX_DRIFT_SEGMENTS spoken lines (riskiest first)
are translated, back-translated blind, and Gemini judges whether the risk changed.

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

from app import config, events, gemini, media, metrics, provisions, taxonomy
from app.schemas import (
    RISK_ORDER, Category, CountryReport, Drift, DriftCheck, Finding, Inventory, OverallRisk, Risk,
    RuleCard, RulesStatus, Segment, SyntheticCheck, Uncovered,
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
    english: str = ""
    leaves: list[str] = []
    idioms: list[str] = []
    synthetic_signals: list[str] = []


class StageADraft(BaseModel):
    """What Gemini returns for stage A; duration comes from ffprobe, not the model."""
    languages: list[str] = []
    segments: list[SegmentDraft] = []
    synthetic: SyntheticCheck | None = None


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
- english: a faithful English translation of the transcript (or, if there is no speech, \
of the on-screen text) when it is not already English; "" when it is English or empty. \
Translate meaning plainly; do not soften threats, slurs or claims.
- leaves: zero or more ids from the content taxonomy below that the segment might \
plausibly touch, most specific first (e.g. a casino promo code is "gambling.promotion", \
not "drugs.promotion"; "this drink cures diabetes" is "health.cure_claim"). A demand \
for money or an act under threat ("pay me or I will ...") is "harassment.extortion" and \
"harassment.threats"; a threat to publish or send someone's intimate or private images \
unless they pay or comply is "harassment.sextortion", "harassment.extortion" and \
"harassment.threats" (add "sexual.non_consensual_imagery" too only if such images are \
shown or said to have already been shared). A false alarm or rumour that could cause \
public fear or panic (e.g. claims of poisoned supplies, attacks or disasters, urging people \
to act or forward it) is "misinfo.public_order" and "hate.public_tranquillity". Use the ids \
exactly as written. Leave empty \
for clearly neutral segments.
- idioms: figures of speech or slang in the speech that could read as literal violence, \
crime, insult or threat if translated word for word (e.g. "this offer is a bomb", \
"it's a killer deal"), quoted verbatim; [] if none.
- synthetic_signals: signs of AI generation seen or heard in this segment (see below); [] if none.

Also return "synthetic", a check for signs that the video was AI-generated or synthetically \
altered. Describe evidence, never guess; when unsure, say less:
- signals: concrete observations, each saying what and where, e.g. "0:04 the hand has six \
fingers", "0:10 the sign's text is garbled", "lip movements do not match the speech", \
"lighting on the face does not match the background", "the face morphs between frames", \
"no blinking for 12 seconds", "the voice sounds synthetic or cloned (flat, metallic, \
no breaths)", or a watermark, caption or spoken line saying the content is AI-made (e.g. \
"AI generated", "Made with AI", Sora, Veo, Runway, Pika, Kling, Synthesia, HeyGen, \
ElevenLabs, Midjourney). [] if none.
- likelihood: "none" (no sign), "low" (one weak or ambiguous sign), "medium" (several clear \
signs, or the video says it is AI-made), "high" (unmistakable artefacts throughout). \
Default to "none" or "low" when unsure.
- label_present: true only if the video itself shows or says, clearly, that it is \
AI-generated or synthetic; label_text: that disclosure verbatim ("" if none).
- depicts_real_person: true only if an identifiable real public figure (politician, \
celebrity, official) appears to be shown or voiced, e.g. named on screen or in the speech, \
or unmistakable; person_note: who, how they appear and what tells you. Never name a \
private person, and never guess an identity from looks alone.
- voice_only: true when every signal is about the voice alone (e.g. text-to-speech \
narration over plain text or stock visuals) and no image, face or footage looks \
AI-made or altered; false otherwise.

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
        hedge_after=config.video_hedge_after(duration_s),
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
        synthetic=with_metadata(draft.synthetic or SyntheticCheck(), media.ai_metadata_hints(video_path)),
    )


_LIKELIHOOD = ["none", "low", "medium", "high"]


def with_metadata(check: SyntheticCheck, hints: list[str]) -> SyntheticCheck:
    """Add file-metadata hints (an AI tool named in the encoder/comment/handler tags): a
    tool writing its own name into the file is a clear sign, so at least "medium"."""
    if not hints:
        return check
    level = max(_LIKELIHOOD.index(check.likelihood), _LIKELIHOOD.index("medium"))
    return check.model_copy(update={"metadata_signals": hints, "likelihood": _LIKELIHOOD[level],
                                    "signals": [*check.signals, *hints]})


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
    return Inventory(duration_s=round(duration, 2), languages=["en"], segments=segments,
                     synthetic=SyntheticCheck())



def inventory_events(inv: Inventory) -> None:
    """Feed events for what stage A saw and heard: one per segment with a claim or a
    taxonomy leaf, then a summary."""
    for s in inv.segments:
        if not (s.leaves or s.claims):
            continue
        if s.transcript.strip():
            what = f"heard {events.quote(s.transcript)}"
        elif s.on_screen_text.strip():
            what = f"saw on-screen text {events.quote(s.on_screen_text)}"
        else:
            what = f"saw {events.quote(s.visuals)}"
        tags = f" · tagged {', '.join(taxonomy.label(x) for x in s.leaves)}" if s.leaves else ""
        events.emit("watch", f"Watched {events.span(s.start_s, s.end_s)} · {what}{tags}",
                    stage="inventory", ref=events.span(s.start_s, s.end_s))
    synthetic_event(inv)
    tagged = sum(1 for s in inv.segments if s.leaves)
    langs = ", ".join(inv.languages) or "none"
    speech = "yes" if has_speech(inv) else "no"
    events.emit("info", f"Inventory built: {len(inv.segments)} moments, {tagged} sensitive · "
                f"languages: {langs} · speech: {speech}", stage="inventory")


def synthetic_event(inv: Inventory) -> None:
    syn = inv.synthetic
    if syn is None:
        return
    signs = "; ".join(syn.signals[:3]) + (f" (+{len(syn.signals) - 3} more)" if len(syn.signals) > 3 else "")
    text = (f"Looked for signs of AI generation: likelihood {syn.likelihood}"
            + (f" · {signs}" if signs else " · no sign seen")
            + (f" · AI label found: {events.quote(syn.label_text, 60)}" if syn.label_present
               else " · no AI label")
            + (" · appears to show or voice a real public figure" if syn.depicts_real_person else ""))
    events.emit("ai", text, stage="ai")


def finalize(rep: CountryReport, inventory: Inventory | None) -> None:
    """Deterministic steps after a country's assessment and its critic (in place, before
    it is reported): severity floors (app/floors.py), then the AI-content check's
    consequence for this country (app/synthetic.py)."""
    from app import floors, synthetic

    if inventory is not None:
        floors.apply(rep, inventory)
        synthetic.apply(rep, inventory)


def country_done_event(rep: CountryReport) -> None:
    name = config.COUNTRIES[rep.country].name
    if rep.overall_risk == "not_checked":
        events.emit("warn", f"{name}: not checked, no verified law is loaded.", stage="country",
                    country=rep.country)
        return
    n, u = len(rep.findings), len(rep.uncovered)
    events.emit("info", f"{name}: {rep.badge.replace('_', ' ')} · {n} finding{'s' * (n != 1)}"
                + (f", {u} to check manually" if u else ""), stage="country", country=rep.country)


# ================================================================ rule cards

_dryrun_lock = threading.Lock()


def load_cards(code: str) -> tuple[list[RuleCard], RulesStatus]:
    """Cards derived from provisions if the country has any (data/provisions),
    else a rule-card file if present, else dry-run cards (generated on demand).
    None at all when the rule-card source law is not verified against its official source.

    Drafted provisions/cards carry needs_review=true; a country is only
    "reviewed" once every card has been signed off.
    """
    country = config.COUNTRIES[code]
    if provisions.has_provisions(code):
        cards = provisions.cards_from_provisions(code)
        status = "draft/unreviewed" if any(c.needs_review for c in cards) else "reviewed"
        return cards, status
    legacy = provisions.legacy_law(code)
    if legacy is None or not legacy.source_verified:
        log.warning("%s: rule-card source not verified, no cards", code)
        return [], "draft/unreviewed"
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
by this law". Never stretch a card to make something fit. Never say content is safe, \
cleared, permitted or carries no liability: these cards are not the whole law.
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


UNVERIFIED_NOTE = ("This country's law could not be verified against its official source, "
                   "so nothing was checked or cited for it.")


def worst_risk(findings: list[Finding]) -> Risk:
    return max((f.risk for f in findings), key=RISK_ORDER.__getitem__, default="green")


def overall_risk(findings: list[Finding], checked: bool) -> OverallRisk:
    """Green only when the country was checked and nothing was found."""
    return worst_risk(findings) if checked else "not_checked"


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
    if not cards:  # e.g. the law's source is not verified: nothing may be cited
        raw = []
    else:
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
        overall_risk=overall_risk(findings, checked=bool(cards)),
        findings=findings,
        rules_status=rules_status,
        risk_framing=_risk_framing(findings),
        cited_cards=[by_id[cid] for cid in cited],
        dropped_findings=len(raw) - len(findings),
        uncovered=uncovered,
        notes=[] if cards else [UNVERIFIED_NOTE],
    )


def stage_b_event(report: CountryReport, n_cards: int) -> None:
    if not n_cards:
        return
    n = len(report.findings)
    dropped = f" · {report.dropped_findings} suggestion(s) dropped by the citation check" \
        if report.dropped_findings else ""
    events.emit("law", f"Checked the moments against {n_cards} rule cards of "
                f"{config.COUNTRIES[report.country].law}: {n} finding{'s' * (n != 1)}{dropped}",
                stage="country", country=report.country)


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
        prompt, list[Finding], model=config.SC_MODEL_SMART, system_instruction=STAGE_B_SYSTEM,
        temperature=config.JUDGE_TEMPERATURE)
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


class _Translation(BaseModel):
    index: int
    language: str
    text: str


class _Judgement(BaseModel):
    index: int
    drift: Drift
    explanation: str


TRANSLATE_SYSTEM = """You are a professional subtitle translator. Translate each item \
faithfully and naturally, the way a dubbing studio would, keeping tone, idioms and \
jokes as a native speaker would render them. Do not soften or sharpen anything. \
Return one output per input index (and per requested language)."""
# Translating is not where the judgement happens: low thinking is ~3x faster
# (median 8.3s -> 2.5s per call, measured); the judge keeps the default.
TRANSLATE_THINKING = "low"

JUDGE_SYSTEM = """You review dubbed lines for a content-compliance tool. For each item \
you get the original line, its translation (language given) and an independent \
back-translation, plus the risk categories already flagged for that moment and any idioms \
in the line (figures of speech that may have been rendered literally).

drift = "increased" if the translated version is likely to read as MORE offensive, \
inflammatory, defamatory or otherwise riskier to that country's audience than the \
original (e.g. a joke that becomes an insult, an idiom that becomes literal violence); \
"reduced" if clearly less risky; otherwise "none". explanation: one or two sentences \
naming the exact words that shifted."""


def _overlaps(f: Finding, s: Segment) -> bool:
    return (f.start_s < s.end_s and s.start_s < f.end_s) or f.start_s == s.start_s


def has_speech(inventory: Inventory) -> bool:
    return any(s.transcript.strip() for s in inventory.segments)


def flagged_segments(inventory: Inventory, report: CountryReport | None = None) -> list[Segment]:
    """Speech segments for the drift check, riskiest first, capped at MAX_DRIFT_SEGMENTS.

    1. segments overlapping one of the country's findings (with no report yet:
       segments stage A tagged with a taxonomy leaf, which every finding needs);
    2. segments with an idiom that could turn literal in translation;
    3. the rest of the speech, so a country with no findings is still checked:
       segments with a taxonomy leaf, then with a claim, then in order.
    """
    speech = [s for s in inventory.segments if s.transcript.strip()]
    if report is None:
        flagged = [s for s in speech if s.leaves]
    else:
        flagged = [s for s in speech if any(_overlaps(f, s) for f in report.findings)]
    idioms = [s for s in speech if s.idioms and s not in flagged]
    rest = sorted((s for s in speech if s not in flagged and s not in idioms),
                  key=lambda s: (not s.leaves, not s.claims, s.start_s))
    return (flagged + idioms + rest)[: config.MAX_DRIFT_SEGMENTS]


# Stage A's ISO 639-1 speech codes that are already the drift language, so translating
# the line into it means nothing. Only unambiguous codes: spoken "zh" may be Cantonese or
# Hokkien, which still need the Mandarin check.
SAME_LANGUAGE_CODES = {
    "Hindi": {"hi"}, "Malay": {"ms"}, "Tamil": {"ta"}, "Urdu": {"ur"},
    "Mandarin Chinese": {"cmn", "zh-cn", "zh-hans", "zh-sg"},
}


def already_in(seg: Segment, language: str) -> bool:
    """Whether the spoken line is already in `language` (no translation check needed)."""
    code = seg.language.strip().lower().replace("_", "-")
    return code in SAME_LANGUAGE_CODES.get(language, ()) or code == language.casefold()


def _clock(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def same_language_note(inventory: Inventory, country: str,
                       report: CountryReport | None = None) -> str:
    """Says which lines skipped a drift language because they are already spoken in it."""
    segs = flagged_segments(inventory, report)
    parts = []
    for lang in config.COUNTRIES[country].drift_languages:
        same = [s for s in segs if already_in(s, lang)]
        if not same:
            continue
        where = "" if len(same) == len(segs) else " for " + ", ".join(
            f"{_clock(s.start_s)}–{_clock(s.end_s)}" for s in same)
        parts.append(f"Already in {lang}: translation check not needed{where}.")
    return " ".join(parts)


def _categories_at(report: CountryReport | None, seg: Segment) -> list[str]:
    if report is None:
        return sorted({c.value for c in seg.categories if c != Category.other})
    return sorted({f.category.value for f in report.findings if _overlaps(f, seg)})


def translation_check(
    inventory: Inventory, country: str, languages: list[str] | tuple[str, ...] | str, *,
    report: CountryReport | None = None, mock: bool | None = None, mock_increase: bool = False,
) -> list[DriftCheck]:
    """Translate -> blind back-translate -> judge, for one country and all its languages:
    three calls whatever the number of languages. `report` (if known) orders the lines
    by its findings; without it stage A's leaves stand in for them. A line already
    spoken in a language is not translated into it (`same_language_note` says so);
    every other language is still checked."""
    mock = config.mock_mode() if mock is None else mock
    languages = [languages] if isinstance(languages, str) else list(languages)
    segs = flagged_segments(inventory, report)
    needed = {i: [lang for lang in languages if not already_in(s, lang)] for i, s in enumerate(segs)}
    if not any(needed.values()):
        return []
    if mock:
        return [d for n, lang in enumerate(languages)
                for d in _mock_drift([s for i, s in enumerate(segs) if lang in needed[i]], lang,
                                     increase_first=mock_increase and n == 0)]

    law = config.COUNTRIES[country]
    hedge = config.text_hedge_after(inventory.duration_s)
    items = [{"index": i, "text": s.transcript, "languages": needed[i]}
             for i, s in enumerate(segs) if needed[i]]
    raw = gemini.generate_json(
        f"Translate every item into each of the languages listed for it ({', '.join(languages)}) "
        f"for viewers in {law.name}. Write each language name exactly as given.\n"
        f"{json.dumps(items, ensure_ascii=False)}",
        list[_Translation], model=config.SC_MODEL_FAST, system_instruction=TRANSLATE_SYSTEM,
        thinking=TRANSLATE_THINKING, hedge_after=hedge)
    wanted = {lang.casefold(): lang for lang in languages}
    translated: dict[int, tuple[int, str, str]] = {}  # k -> (segment index, language, text)
    seen: set[tuple[int, str]] = set()
    for item in raw if isinstance(raw, list) else []:
        try:
            t = _Translation.model_validate(item)
        except ValidationError:
            continue
        lang = wanted.get(t.language.strip().casefold())
        if lang and lang in needed.get(t.index, ()) and (t.index, lang) not in seen:
            seen.add((t.index, lang))
            translated[len(translated)] = (t.index, lang, t.text)
    if not translated:
        return []
    # Back-translate without the original (or the language) in context so the model can't copy it.
    back = _by_index(gemini.generate_json(
        "Translate into English.\n" + json.dumps(
            [{"index": k, "text": text} for k, (_, _, text) in translated.items()], ensure_ascii=False),
        list[_Text], model=config.SC_MODEL_FAST, system_instruction=TRANSLATE_SYSTEM,
        thinking=TRANSLATE_THINKING, hedge_after=hedge), _Text)
    judge_items = [
        {"index": k, "language": lang, "original": segs[i].transcript, "translation": text,
         "back_translation": back[k].text, "flagged_categories": _categories_at(report, segs[i]),
         "idioms": segs[i].idioms}
        for k, (i, lang, text) in translated.items() if k in back
    ]
    if not judge_items:
        return []
    judged = _by_index(gemini.generate_json(
        f"Audience: {law.name}.\n{json.dumps(judge_items, ensure_ascii=False)}",
        list[_Judgement], model=config.SC_MODEL_SMART, system_instruction=JUDGE_SYSTEM,
        hedge_after=hedge), _Judgement)
    out = [
        DriftCheck(start_s=segs[i].start_s, end_s=segs[i].end_s, language=lang,
                   original=segs[i].transcript, translated=text, back_translated=back[k].text,
                   drift=judged[k].drift, explanation=judged[k].explanation)
        for k, (i, lang, text) in translated.items() if k in back and k in judged
    ]
    return sorted(out, key=lambda d: (languages.index(d.language), d.start_s))


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


NO_SPEECH = "No speech to translate."


async def run_drift(
    inventory: Inventory, countries: list[str], *,
    reports: dict[str, CountryReport] | None = None, mock: bool | None = None,
) -> tuple[dict[str, list[DriftCheck]], dict[str, str], list[str]]:
    """Drift checks for every country with a drift language, findings or not: a line
    that is fine in English can become risky in translation. Needs only the
    inventory, so it can run while the countries are being assessed (`reports`,
    if given, orders the lines by findings). Returns (checks, notes, errors) per country."""
    mock = config.mock_mode() if mock is None else mock
    todo = [cc for cc in countries if config.COUNTRIES[cc].drift_languages]
    if not has_speech(inventory):
        return {}, {cc: NO_SPEECH for cc in todo}, []
    rep = (lambda cc: (reports or {}).get(cc))
    # Mock: exactly one deterministic "increased" example, on the first eligible country.
    first = next((cc for cc in todo if flagged_segments(inventory, rep(cc))), None)
    sem = asyncio.Semaphore(max(1, config.GEMINI_CONCURRENCY))

    async def one(cc: str):
        async with sem:
            with metrics.stage(f"translation:{cc}"):
                out = await asyncio.to_thread(
                    translation_check, inventory, cc, config.COUNTRIES[cc].drift_languages,
                    report=rep(cc), mock=mock, mock_increase=cc == first)
        translation_events(cc, out)
        return out

    results = await asyncio.gather(*(one(cc) for cc in todo), return_exceptions=True)
    if mock and todo:
        await asyncio.sleep(config.mock_delay_s())
    checks, errors = {}, []
    for cc, res in zip(todo, results):
        if isinstance(res, BaseException):
            log.error("translation check failed for %s: %s", cc, type(res).__name__)
            errors.append(f"translation {cc}: {public_error(res)}")
            continue
        checks[cc] = res
    notes = {cc: note for cc in todo if (note := same_language_note(inventory, cc, rep(cc)))}
    return checks, notes, errors


def translation_events(country: str, checks: list[DriftCheck]) -> None:
    """One feed event per language: lines checked and meaning shifts found."""
    for lang in config.COUNTRIES[country].drift_languages:
        mine = [d for d in checks if d.language == lang]
        if not mine:
            continue
        up = sum(d.drift == "increased" for d in mine)
        down = sum(d.drift == "reduced" for d in mine)
        shifts = (f"{up} reads riskier" if up else "no line reads riskier") + (
            f", {down} reads milder" if down else "")
        events.emit("warn" if up else "translate",
                    f"{lang} for {config.COUNTRIES[country].name}: {len(mine)} line"
                    f"{'s' * (len(mine) != 1)} translated, back-translated and compared · {shifts}",
                    stage="translation", country=country)


def attach_drift(reports: dict[str, CountryReport], checks: dict[str, list[DriftCheck]],
                 notes: dict[str, str]) -> None:
    for cc, rep in reports.items():
        rep.translation.extend(checks.get(cc, []))
        rep.translation_note = notes.get(cc, rep.translation_note)


async def run_translation(
    inventory: Inventory, reports: dict[str, CountryReport], *, mock: bool | None = None,
) -> list[str]:
    """Fill report.translation for every country with a drift language. Returns errors."""
    checks, notes, errors = await run_drift(inventory, list(reports), reports=reports, mock=mock)
    attach_drift(reports, checks, notes)
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
                stage_b_event(report, len(cards))
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
