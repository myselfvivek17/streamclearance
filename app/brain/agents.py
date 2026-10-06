"""The LLM agents of the brain: country assessor and critic (Google ADK `Agent`s).

Assessor: one per country, with tools `lookup_provisions` and `search_law`;
returns one MomentAssessment per (sensitive moment, leaf). If ADK tool calling
fails, it falls back to: retrieval in Python -> one
JSON call with the retrieved provisions in the prompt. The fallback is noted
in the report.

Critic: one pass over the verified findings with each cited provision's
verbatim text and exceptions; may keep, downgrade or remove (with a reason).
If it fails, findings stay as they are and a note says so.

Mock mode: deterministic stand-ins, no network.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Literal

from pydantic import BaseModel, ValidationError

from app import config, gemini, metrics, provisions, retrieval
from app.brain.models import adk_model, count_call, log_usage
from app.brain.tools import make_tools, provision_view
from app.schemas import Inventory, Risk

log = logging.getLogger(__name__)


class MomentAssessment(BaseModel):
    """Assessor output for one (moment, leaf) in one country."""
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


class CriticVerdict(BaseModel):
    index: int
    action: Literal["keep", "downgrade", "remove"] = "keep"
    risk: Risk | None = None  # new risk when downgrading
    reason: str = ""


def moments(inventory: Inventory) -> list[dict]:
    """Sensitive moments: segments stage A tagged with at least one taxonomy leaf."""
    return [
        {"start_s": s.start_s, "end_s": s.end_s, "leaves": s.leaves, "transcript": s.transcript,
         "claims": s.claims, "visuals": s.visuals, "on_screen_text": s.on_screen_text}
        for s in inventory.segments if s.leaves
    ]


# ================================================================ ADK plumbing

async def run_agent(agent, message: str) -> str:
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


def _parse_list(text: str, model: type[BaseModel]) -> list:
    # Prefer the first value holding objects: prose before the answer may quote
    # a bracketed list of ids, which is not the answer.
    values = list(gemini.iter_json(text))
    raw = next((v for v in values if isinstance(v, dict)
                or (isinstance(v, list) and any(isinstance(x, dict) for x in v))),
               values[0] if values else None)
    if raw is None:
        raise ValueError("no JSON found in model output")
    if isinstance(raw, dict):
        lists = [v for v in raw.values() if isinstance(v, list)]
        raw = lists[0] if len(lists) == 1 else [raw]
    out = []
    for item in raw:
        try:
            out.append(model.model_validate(item))
        except ValidationError as e:
            log.warning("dropping malformed %s: %s", model.__name__, e.errors()[0]["type"])
    return out


# ================================================================ assessor

ASSESSOR_INSTRUCTION = """You are the legal assessor for ONE country ({country}) in a tool \
that warns online video creators which moments of their video may be taken down, \
geo-blocked or penalised there.

You get the video's sensitive moments, each tagged with content-taxonomy leaves. For EVERY \
moment and EVERY leaf of it:
1. Call lookup_provisions(country="{country}", leaves=[...]) to fetch the provisions of \
{country}'s loaded laws that target that leaf. If nothing fits, you may call \
search_law(country="{country}", query=...) once with words from the moment.
2. Decide whether a returned provision's verbatim_text REALLY covers this moment. Sharing a \
broad topic is not enough (a dangerous-drugs provision does not cover a casino promotion). \
Never cite an id you did not get from a tool. If none fits, covered_by = [] ("not covered \
by any loaded law") - that is a valid, honest answer.
3. Check the provision's exceptions (e.g. education, awareness, news, government \
advertisement). If one plausibly applies, set exception_applies=true and say why.

Output ONLY a JSON array, one object per (moment, leaf), no prose:
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
topic is not enough). If none fits, covered_by = []. Check each provision's exceptions. \
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


def _assess_prompt(country: str, ms: list[dict]) -> str:
    return (f"Country: {country}. Video moments (JSON):\n{json.dumps(ms, ensure_ascii=False)}\n\n"
            "Assess every (moment, leaf) and reply with the JSON array only.")


async def assess(inventory: Inventory, country: str, *, mock: bool) -> tuple[list[MomentAssessment], list[str]]:
    """Returns (assessments, notes)."""
    if mock:
        return mock_assess(inventory, country), []
    ms = moments(inventory)
    if not ms:
        return [], []
    from google.adk import Agent

    lookup_provisions, search_law = make_tools(country)
    fallbacks = retrieval.track_fallbacks()  # law-search errors -> a report note
    agent = Agent(
        name=f"country_assessor_{country}", model=adk_model(),
        instruction=ASSESSOR_INSTRUCTION.format(country=country),
        tools=[lookup_provisions, search_law],
        before_model_callback=count_call, after_model_callback=log_usage,
    )
    try:
        text = await run_agent(agent, _assess_prompt(country, ms))
        out = _parse_list(text, MomentAssessment)
        if not out:
            raise ValueError("no assessments in the agent's reply")
        return out, fallbacks
    except gemini.GeminiUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 - fall back, and say so in the report
        log.warning("%s: ADK assessor failed (%s); falling back to retrieval + one JSON call",
                    country, type(e).__name__)
        note = (f"Assessor fallback: ADK tool-calling run failed ({type(e).__name__}); "
                "provisions were retrieved in Python and assessed in one JSON call.")
        out = await _fallback_assess(country, ms)
        return out, [note, *fallbacks]


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
    import asyncio

    items = await asyncio.to_thread(  # path B may query RAG Engine: off the event loop
        lambda: [{**m, "candidate_provisions": candidates(country, m)} for m in ms])
    prompt = (f"Country: {country}.\nMoments with their candidate provisions (JSON):\n"
              f"{json.dumps(items, ensure_ascii=False)}\n\nReturn the assessments as a JSON array.")
    raw = await asyncio.to_thread(
        gemini.generate_json, prompt, list[MomentAssessment], model=config.SC_MODEL_SMART,
        system_instruction=FALLBACK_SYSTEM)
    return _parse_list(json.dumps(raw), MomentAssessment) if raw else []


# ================================================================ critic

CRITIC_INSTRUCTION = """You are the critic in a content-compliance tool. Each finding \
claims a video moment is covered by one or more law provisions; you get the moment and the \
provisions' verbatim text and exceptions. For each finding decide:
- "keep": the provision text really covers this moment;
- "downgrade": covered, but arguable or an exception (education, news, satire, awareness, \
government advertisement...) may apply -> give the lower "risk" ("amber" or "green");
- "remove": the provision text does not actually cover this moment.
Be strict about real coverage, not about tone. Output ONLY a JSON array: \
[{"index": int, "action": "keep"|"downgrade"|"remove", "risk": "amber"|"green"|null, \
"reason": str}] with one item per finding."""


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
