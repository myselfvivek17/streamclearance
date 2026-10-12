"""The LLM agents of the brain: country assessor and critic (Google ADK `Agent`s).

Assessor: one per country. Its prompt carries the exact-lookup provisions for
every moment (`prefetch`); tools `lookup_provisions` and `search_law` answer at
most SC_ASSESSOR_MAX_TOOL_CALLS calls. Returns one MomentAssessment per (sensitive moment, leaf). If ADK tool calling
fails, it falls back to: retrieval in Python -> one
JSON call with the retrieved provisions in the prompt. The fallback is noted
in the report.

Critic: one pass over the verified findings with each cited provision's
verbatim text and exceptions; may keep, downgrade or remove (with a reason).
If it fails, findings stay as they are and a note says so.

Mock mode: deterministic stand-ins, no network.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ValidationError, field_validator

from app import config, events, gemini, metrics, provisions, retrieval, taxonomy
from app.brain.models import adk_model, agent_config, count_call, log_usage
from app.brain.tools import cite, make_tools, provision_view
from app.schemas import Inventory, Risk

log = logging.getLogger(__name__)


class MomentAssessment(BaseModel):
    """Assessor output for one (moment, leaf) in one country. Unknown keys are ignored."""
    start_s: float
    end_s: float
    leaf: str
    covered_by: list[str] = []  # provision ids; [] = not covered by any loaded law
    risk: Risk = "amber"
    exception_applies: bool = False
    exception_reason: str = ""
    suggested_fix: str = ""
    rationale: str = ""
    critic_note: str = ""  # set by us from the critic's verdict

    @field_validator("covered_by", mode="before")
    @classmethod
    def _ids(cls, v):
        if v is None:
            return []
        return [v] if isinstance(v, str) else v

    @field_validator("risk", mode="before")
    @classmethod
    def _risk(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class CriticVerdict(BaseModel):
    index: int
    action: Literal["keep", "downgrade", "remove"] = "keep"
    risk: Risk | None = None  # new risk when downgrading
    reason: str = ""

    @field_validator("action", "risk", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


@dataclass
class Assessed:
    """The assessor's answer for one country, and how it was reached."""
    assessments: list[MomentAssessment]
    notes: list[str] = field(default_factory=list)
    method: Literal["agent", "fallback"] = "agent"
    fallback_reason: str | None = None  # FALLBACK_REASONS code


# Why the assessor agent's run was replaced by the simplified check.
FALLBACK_REASONS = {
    "no_json": "the agent's reply held no JSON",
    "empty_reply": "the agent replied with an empty list",
    "timeout": "the agent did not answer in time",
    "no_valid_items": "the agent's reply held no valid assessment",
    "agent_error": "the agent run failed",
}


def moments(inventory: Inventory) -> list[dict]:
    """Sensitive moments: segments stage A tagged with at least one taxonomy leaf."""
    return [
        {"start_s": s.start_s, "end_s": s.end_s, "leaves": s.leaves, "transcript": s.transcript,
         "claims": s.claims, "visuals": s.visuals, "on_screen_text": s.on_screen_text}
        for s in inventory.segments if s.leaves
    ]


# ================================================================ ADK plumbing

def _rate_limited(e: BaseException) -> bool:
    text = f"{type(e).__name__} {str(e)[:300]}"
    return "RESOURCE_EXHAUSTED" in text or "ResourceExhausted" in text or " 429" in text


async def run_agent(agent, message: str, *, retries: int | None = None) -> str:
    """run_agent_once, retried with backoff on rate limits (429), like app/gemini.py
    `with_retry` for direct calls; a retry starts a fresh session."""
    retries = config.GEMINI_MAX_RETRIES if retries is None else retries
    for attempt in range(retries + 1):
        try:
            return await run_agent_once(agent, message)
        except gemini.GeminiUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - ADK surfaces several types
            if not _rate_limited(e) or attempt == retries:
                raise
            delay = min(30.0, 2.0 * 2**attempt) + random.uniform(0, 1)
            log.warning("%s: rate limited, retry %d/%d in %.1fs", agent.name, attempt + 1, retries, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def run_agent_once(agent, message: str) -> str:
    """_run_agent with a deadline (SC_AGENT_TIMEOUT_S): a hung model request must not hang
    the job. Raises TimeoutError."""
    return await asyncio.wait_for(_run_agent(agent, message), timeout=config.AGENT_TIMEOUT_S)


async def _run_agent(agent, message: str) -> str:
    """Run one ADK agent to completion in its own in-memory session; return its final text."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="streamclearance")
    session = await runner.session_service.create_session(
        app_name="streamclearance", user_id="brain", session_id=uuid.uuid4().hex)
    text = ""
    async for ev in runner.run_async(
            user_id="brain", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)])):
        if ev.error_message:
            raise RuntimeError(f"{agent.name}: {ev.error_message}")
        if ev.content and ev.content.parts and not ev.partial:
            chunk = "".join(p.text for p in ev.content.parts if p.text and not p.thought)
            if chunk:
                text = chunk  # keep the last text answer (after tool calls)
    return text


# Wrapper keys models put around the answer list, e.g. {"assessments": [...]}.
WRAPPER_KEYS = ("assessments", "assessment", "items", "results", "findings", "verdicts", "moments",
                "data", "output", "answer", "response")


def _items(value, fields: set[str], inherited: dict | None = None, depth: int = 0) -> list:
    """Flatten a model reply into candidate items for a list answer. Tolerates: a nested
    list (one list per moment), JSON objects encoded as strings, a wrapper object
    ({"assessments": [...]}), items grouped under their moment ({"start_s": 1, "end_s": 2,
    "assessments": [{"leaf": ...}]}: the inner items inherit the times), and a dict keyed
    by index or moment ({"0": {...}}). Anything else is returned as is, and dropped (with
    its field names logged) by the caller."""
    inherited = inherited or {}
    if depth > 5:
        return [value]
    if isinstance(value, str):
        try:
            return _items(json.loads(value), fields, inherited, depth + 1)
        except ValueError:
            return [value]
    if isinstance(value, list):
        return [x for v in value for x in _items(v, fields, inherited, depth + 1)]
    if not isinstance(value, dict):
        return [value]
    wrapped = [k for k, v in value.items() if k.lower() in WRAPPER_KEYS and isinstance(v, (list, dict))]
    if wrapped:  # a wrapper, or a moment holding its items
        moment = value.get("moment") if isinstance(value.get("moment"), dict) else value
        times = {k: moment[k] for k in ("start_s", "end_s") if k in moment}
        return [x for k in wrapped for x in _items(value[k], fields, {**inherited, **times}, depth + 1)]
    if fields & set(value):
        return [{**inherited, **value}]
    if value and all(isinstance(v, dict) for v in value.values()):  # keyed by index / moment
        return [x for v in value.values() for x in _items(v, fields, inherited, depth + 1)]
    return [value]


def _parse_list(text: str, model: type[BaseModel]) -> list:
    # Prefer the first value holding objects: prose before the answer may quote
    # a bracketed list of ids, which is not the answer.
    values = list(gemini.iter_json(text))

    def holds_objects(v, depth=0) -> bool:
        if isinstance(v, dict):
            return True
        return isinstance(v, list) and depth < 3 and any(holds_objects(x, depth + 1) for x in v)

    raw = next((v for v in values if holds_objects(v)), values[0] if values else None)
    if raw is None:
        raise ValueError("no JSON found in model output")
    fields = set(model.model_fields)
    out = []
    for item in _items(raw, fields):
        try:
            out.append(model.model_validate(item))
        except ValidationError as e:
            err = e.errors()[0]
            # field names and the error type only: values are model output
            where = ".".join(str(x) for x in err["loc"]) or type(item).__name__
            log.warning("dropping malformed %s: %s at %s", model.__name__, err["type"], where)
    return out


# ================================================================ assessor

ASSESSOR_INSTRUCTION = """You are the legal assessor for ONE country ({country}) in a tool \
that warns online video creators which moments of their video may be taken down, \
geo-blocked or penalised there.

You get the video's sensitive moments, each tagged with content-taxonomy leaves, and \
the provisions of {country}'s loaded laws that target those leaves (already looked up for \
you: each moment lists its candidate_provision_ids; the provisions' full text is under \
PROVISIONS). For EVERY moment and EVERY leaf of it:
1. Use the given provisions. Only if none of them fits a moment, you may call \
search_law(country="{country}", query=...) with words from the moment; at most \
{max_calls} tool calls in total, then answer with what you have.
2. Decide whether a returned provision's verbatim_text REALLY covers this moment. Sharing a \
broad topic is not enough (a dangerous-drugs provision does not cover a casino promotion). \
Never cite an id that was not given to you or returned by a tool. If none fits, covered_by = [] ("not covered \
by any loaded law") - that is a valid, honest answer. Its rationale says only which loaded \
provisions were considered and why they do not fit; never say the creator is safe, not \
liable, cleared or permitted (other laws may still apply).
3. Check the provision's exceptions (e.g. education, awareness, news, government \
advertisement). If one plausibly applies, set exception_applies=true and say why.

Output ONLY a flat JSON array of objects, one object per (moment, leaf): no prose, no \
wrapper object, no nesting or grouping by moment, every object with its own start_s and end_s:
{{"start_s": float, "end_s": float, "leaf": "<one leaf of that moment>", "covered_by": \
["<provision id>", ...], "risk": "red"|"amber"|"green", "exception_applies": bool, \
"exception_reason": str, "suggested_fix": str, "rationale": str}}
risk: red = clearly within the provision; amber = arguable/context-dependent; green = \
covered topic but this moment is fine. rationale: 1-3 sentences tying the words/visuals \
to the provision text; say whether the duty falls on the platform (takedown/geo-block) \
or on the creator/advertiser (offence). suggested_fix: a concrete edit (cut, reword, \
blur, add context) that keeps the creator's intent."""

FALLBACK_SYSTEM = """You are the legal assessor for one country in a tool that warns \
online video creators which moments may be taken down, geo-blocked or penalised. For each \
moment you get the candidate provisions retrieved for it. Cite ONLY those provision ids, \
and only when the provision's verbatim_text really covers the moment (sharing a broad \
topic is not enough). If none fits, covered_by = [] and never say the content is safe, \
cleared, permitted or carries no liability. Check each provision's exceptions. \
Return one item per (moment, leaf). risk: red = clearly within; amber = arguable; green = \
fine. rationale: 1-3 sentences; suggested_fix: a concrete edit."""

_MOCK_RISK: dict[str, Risk] = {"health": "red", "gambling": "red", "drugs": "red",
                               "religion": "amber", "hate": "amber"}


def mock_assess(inventory: Inventory, country: str) -> list[MomentAssessment]:
    """Deterministic: cite the first looked-up provision, else not covered; plus one
    invented citation that the verifier must drop."""
    out = []
    for m in moments(inventory):
        for leaf in m["leaves"]:
            hits = retrieval.lookup(country, [leaf])
            out.append(MomentAssessment(
                start_s=m["start_s"], end_s=m["end_s"], leaf=leaf,
                covered_by=[hits[0].id] if hits else [],
                risk=_MOCK_RISK.get(leaf.split(".")[0], "amber"),
                suggested_fix="(mock) Cut or reword this moment for this market.",
                rationale=(f"(mock) \"{m['transcript'][:60]}\" falls under {hits[0].section} "
                           f"of {hits[0].law_title}." if hits else "(mock) no loaded provision fits."),
            ))
    ms = moments(inventory)
    if ms:
        out.append(MomentAssessment(
            start_s=ms[0]["start_s"], end_s=ms[0]["end_s"], leaf=ms[0]["leaves"][0],
            covered_by=[f"{country}/INVENTED/s999"], risk="red", suggested_fix="-",
            rationale="(mock) hallucinated citation; the verifier must drop this."))
    return out


def prefetch(country: str, ms: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Path A lookup for every moment, done here instead of by the agent: saves one
    model round trip per lookup (each re-sends the whole conversation)."""
    provs: dict[str, dict] = {}
    out = []
    with metrics.timed(f"retrieval:{country}"):
        for m in ms:
            hits = retrieval.lookup(country, m["leaves"])
            for p in hits:
                provs.setdefault(p.id, provision_view(p))
            out.append({**m, "candidate_provision_ids": [p.id for p in hits]})
    laws = sorted({p["law"] for p in provs.values()})
    events.emit("law", f"Looked up {len(provs)} candidate provision{'s' * (len(provs) != 1)} for "
                f"{len(ms)} sensitive moment{'s' * (len(ms) != 1)}"
                + (f" in {'; '.join(laws)}" if laws else " (no loaded provision targets them)"),
                stage="country", country=country)
    return out, provs


def assessment_events(country: str, out: list[MomentAssessment]) -> None:
    by_id = provisions.by_id(country)
    for a in out:
        where = events.span(a.start_s, a.end_s)
        leaf = a.leaf.strip().lower()
        label = taxonomy.label(leaf) if taxonomy.is_leaf(leaf) else "a tagged leaf"
        refs = [cite(by_id[pid]) if pid in by_id else "an unknown id" for pid in a.covered_by]
        text = (f"Assessed {where} {label} → {', '.join(refs)} ({a.risk})" if refs
                else f"Assessed {where} {label} → no loaded provision fits")
        events.emit("law", text, stage="country", country=country, ref=where)


def _assess_prompt(country: str, ms: list[dict], provs: dict[str, dict] | None = None) -> str:
    text = f"Country: {country}. Video moments (JSON):\n{json.dumps(ms, ensure_ascii=False)}\n\n"
    if provs is not None:
        text += f"PROVISIONS (JSON, by id):\n{json.dumps(provs, ensure_ascii=False)}\n\n"
    return text + "Assess every (moment, leaf) and reply with the JSON array only."


class _NoAssessments(ValueError):
    pass


async def assess(inventory: Inventory, country: str, *, mock: bool) -> Assessed:
    if mock:
        out = mock_assess(inventory, country)
        assessment_events(country, out)
        return Assessed(out)
    ms = moments(inventory)
    if not ms:
        return Assessed([])
    from google.adk import Agent

    lookup_provisions, search_law = make_tools(country, max_calls=config.ASSESSOR_MAX_TOOL_CALLS)
    fallbacks = retrieval.track_fallbacks()  # law-search errors -> a report note
    agent = Agent(
        name=f"country_assessor_{country}", model=adk_model(),
        instruction=ASSESSOR_INSTRUCTION.format(country=country,
                                                max_calls=config.ASSESSOR_MAX_TOOL_CALLS),
        tools=[lookup_provisions, search_law],
        generate_content_config=agent_config(),  # the assessor keeps the default thinking
        before_model_callback=count_call, after_model_callback=log_usage,
    )
    try:
        items, provs = await asyncio.to_thread(prefetch, country, ms)
        text = await run_agent(agent, _assess_prompt(country, items, provs))
        try:
            out = _parse_list(text, MomentAssessment)
        except ValueError as e:
            raise _NoAssessments("no_json") from e
        if not out:
            empty = [v for v in gemini.iter_json(text)] in ([[]], [{}])
            raise _NoAssessments("empty_reply" if empty else "no_valid_items")
        assessment_events(country, out)
        return Assessed(out, fallbacks)
    except gemini.GeminiUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 - fall back, and say so in the report and the feed
        reason = str(e) if isinstance(e, _NoAssessments) else \
            "timeout" if isinstance(e, asyncio.TimeoutError) else "agent_error"
        log.warning("%s: ADK assessor failed (%s, %s); falling back to retrieval + one JSON call",
                    country, type(e).__name__, reason)
        note = (f"Simplified check: {FALLBACK_REASONS[reason]} ({type(e).__name__}), so the provisions "
                "were retrieved in code and assessed in one structured call instead of by the agent.")
        events.emit("warn", f"The assessor agent's answer could not be used ({FALLBACK_REASONS[reason]}): "
                    "switched to a simplified check (provisions retrieved in code, one structured call).",
                    stage="country", country=country)
        out = await _fallback_assess(country, ms)
        assessment_events(country, out)
        return Assessed(out, [note, *fallbacks], method="fallback", fallback_reason=reason)


def candidates(country: str, m: dict) -> list[dict]:
    """Path A lookup, topped up with path B search on the moment's words."""
    with metrics.timed(f"retrieval:{country}"):
        hits = retrieval.lookup(country, m["leaves"])
        query = " ".join([m["transcript"], *m["claims"], m["on_screen_text"]])
        for p in retrieval.search(country, query, k=3):
            if p not in hits:
                hits.append(p)
    return [provision_view(p) for p in hits[: retrieval.MAX_RESULTS]]


async def _fallback_assess(country: str, ms: list[dict]) -> list[MomentAssessment]:
    items = await asyncio.to_thread(  # path B may query RAG Engine: off the event loop
        lambda: [{**m, "candidate_provisions": candidates(country, m)} for m in ms])
    prompt = (f"Country: {country}.\nMoments with their candidate provisions (JSON):\n"
              f"{json.dumps(items, ensure_ascii=False)}\n\nReturn the assessments as a JSON array.")
    raw = await asyncio.to_thread(
        gemini.generate_json, prompt, list[MomentAssessment], model=config.SC_MODEL_SMART,
        system_instruction=FALLBACK_SYSTEM, temperature=config.JUDGE_TEMPERATURE)
    return _parse_list(json.dumps(raw), MomentAssessment) if raw else []


# ================================================================ critic

CRITIC_INSTRUCTION = """You are the critic in a content-compliance tool. Each finding \
claims a video moment is covered by one or more law provisions; you get the moment and the \
provisions' verbatim text and exceptions. For each finding decide:
- "keep": the provision text really covers this moment;
- "downgrade": covered, but arguable or an exception (education, news, satire, awareness, \
government advertisement...) may apply -> give the lower "risk" ("amber" or "green");
- "remove": the provision text does not actually cover this moment.
Be strict about real coverage, not about tone. Output ONLY a flat JSON array of objects \
(no wrapper object, no nesting): \
[{"index": int, "action": "keep"|"downgrade"|"remove", "risk": "amber"|"green"|null, \
"reason": str}] with one item per finding."""


# The critic answers keep / downgrade / remove per finding against quoted text: a simple
# structured call, so low thinking (measured on the eval clips; the assessor keeps the default).
CRITIC_THINKING = config.CRITIC_THINKING


def _critic_items(country: str, found: list[MomentAssessment], inventory: Inventory) -> list[dict]:
    by_id = provisions.by_id(country)
    items = []
    for i, a in enumerate(found):
        seg = next((s for s in inventory.segments if s.start_s == a.start_s), None)
        items.append({
            "index": i, "leaf": a.leaf, "risk": a.risk, "rationale": a.rationale,
            "exception_applies": a.exception_applies,
            "moment": {"transcript": seg.transcript, "claims": seg.claims, "visuals": seg.visuals,
                       "on_screen_text": seg.on_screen_text} if seg else {},
            "provisions": [{k: v for k, v in provision_view(by_id[pid]).items()
                            if k in ("id", "law", "section", "verbatim_text", "exceptions")}
                           for pid in a.covered_by if pid in by_id],
        })
    return items


def mock_critic(found: list[MomentAssessment]) -> list[CriticVerdict]:
    """Deterministic: downgrade red findings whose assessor said an exception applies."""
    return [CriticVerdict(index=i, action="downgrade", risk="amber",
                          reason="(mock) an exception may apply")
            if a.exception_applies and a.risk == "red" else CriticVerdict(index=i)
            for i, a in enumerate(found)]


async def critique(found: list[MomentAssessment], inventory: Inventory, country: str, *,
                   mock: bool) -> tuple[list[CriticVerdict], list[str]]:
    if not found:
        return [], []
    if mock:
        return mock_critic(found), []
    from google.adk import Agent

    agent = Agent(name=f"critic_{country}", model=adk_model(), instruction=CRITIC_INSTRUCTION,
                  generate_content_config=agent_config(thinking=CRITIC_THINKING),
                  before_model_callback=count_call, after_model_callback=log_usage)
    try:
        text = await run_agent(agent, json.dumps(_critic_items(country, found, inventory),
                                                 ensure_ascii=False))
        return _parse_list(text, CriticVerdict), []
    except gemini.GeminiUnavailable:
        raise
    except Exception as e:  # noqa: BLE001
        log.warning("%s: critic failed (%s); findings kept unchanged", country, type(e).__name__)
        return [], [f"Critic unavailable ({type(e).__name__}); findings were not re-checked."]


def critic_event(country: str, found: list[MomentAssessment], verdicts: list[CriticVerdict],
                 removed: list[tuple[MomentAssessment, str]], changes: int) -> None:
    """One feed event: what the reviewer (critic) kept, downgraded or dropped, and why."""
    if not found:
        return
    if not verdicts:
        events.emit("warn", "Reviewer unavailable: findings were not re-checked.", stage="country",
                    country=country)
        return
    downgraded = changes - len(removed)
    parts = [f"Reviewer kept {len(found) - len(removed) - downgraded}"]
    if downgraded:
        parts.append(f"downgraded {downgraded}")
    if removed:
        parts.append(f"dropped {len(removed)}")
    reasons = "; ".join(r for _, r in removed if r)[:160]
    events.emit("review", ", ".join(parts) + (f": {reasons}" if reasons else ""), stage="country",
                country=country)


def apply_verdicts(found: list[MomentAssessment], verdicts: list[CriticVerdict]
                   ) -> tuple[list[MomentAssessment], list[tuple[MomentAssessment, str]], int]:
    """Returns (kept, removed with reason, number of changes). Downgrades only lower risk."""
    order = {"green": 0, "amber": 1, "red": 2}
    by_index = {v.index: v for v in verdicts if 0 <= v.index < len(found)}
    kept, removed, changes = [], [], 0
    for i, a in enumerate(found):
        v = by_index.get(i)
        if v is None or v.action == "keep":
            kept.append(a)
        elif v.action == "remove":
            removed.append((a, v.reason))
            changes += 1
        else:
            new = v.risk if v.risk and order[v.risk] < order[a.risk] else a.risk
            changes += new != a.risk
            kept.append(a.model_copy(update={
                "risk": new, "critic_note": f"Critic: {v.reason}" if v.reason else ""}))
    return kept, removed, changes
