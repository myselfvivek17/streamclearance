"""The brain (SC_BRAIN=adk): a Google ADK `Workflow` graph.

    START -> perceiver -> ( assessor_XX -> verifier_XX -> critic_XX )  one chain per country,
                          run in parallel (Workflow.max_concurrency)  -> join -> reporter

- perceiver: existing stage A (tags moments with taxonomy leaves);
- assessor_XX: ADK Agent with lookup_provisions / search_law tools (app/brain/agents.py);
- verifier_XX: plain Python (app/brain/verifier.py);
- critic_XX: ADK Agent, one pass;
- countries without provisions yet fall back to legacy stage B on their rule cards.
Translation drift and the final Report stay in app/jobs.py, as for the legacy path.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app import config, events, metrics, pipeline, provisions, retrieval, taxonomy
from app.brain import agents, verifier
from app.brain.agents import MomentAssessment
from app.schemas import (
    RISK_ORDER, CountryReport, Finding, Inventory, Provision, Uncovered,
)

log = logging.getLogger(__name__)

MAX_OBLIGATIONS = 8  # per country report, most relevant first

StageHook = Callable[[str, str, str], Awaitable[None]]  # (stage key, status, detail)


def _framing(findings: list[Finding]) -> str:
    if not findings:
        return "No moment is covered by a loaded provision of this country's laws."
    if any(f.creator_liability for f in findings):
        return ("At least one cited provision makes the person who publishes the content (creator "
                "or advertiser) liable; see its penalty. Platform takedown or geo-blocking also applies.")
    return ("The cited provisions place duties on platforms, so the practical risk is takedown "
            "or geo-blocking of this video in this country, not liability for the creator.")


def build_report(country: str, inventory: Inventory, kept: list[MomentAssessment],
                 removed: list[tuple[MomentAssessment, str]], v: verifier.Verified,
                 critic_changes: int, notes: list[str]) -> CountryReport:
    by_id = provisions.by_id(country)
    findings: list[Finding] = []
    duties: dict[str, Provision] = {}
    for a in sorted(kept, key=lambda a: -RISK_ORDER[a.risk]):
        if a.risk == "green":
            continue
        cited = [by_id[pid] for pid in a.covered_by]
        obligations = retrieval.obligations(country, [a.leaf])
        for p in obligations:
            duties.setdefault(p.id, p)
        findings.append(Finding(
            start_s=a.start_s, end_s=a.end_s, category=taxonomy.category_of(a.leaf),
            rule_card_ids=a.covered_by, risk=a.risk, suggested_fix=a.suggested_fix,
            rationale=a.rationale, leaf=a.leaf,
            creator_liability=any(p.duty_holder in ("creator", "advertiser", "both") for p in cited),
            exception_applies=a.exception_applies, exception_reason=a.exception_reason,
            critic_note=a.critic_note, obligation_ids=[p.id for p in obligations],
        ))
    findings.sort(key=lambda f: (f.start_s, -RISK_ORDER[f.risk]))
    platform_duties = list(duties.values())[:MAX_OBLIGATIONS]
    shown = {p.id for p in platform_duties}
    for f in findings:
        f.obligation_ids = [pid for pid in f.obligation_ids if pid in shown]

    assessed = {(a.start_s, a.leaf) for a in kept}  # covered (incl. green) moments
    why = {(a.start_s, a.leaf): a.rationale for a in v.not_covered}
    why |= {(a.start_s, a.leaf): f"Critic removed the citation: {r}" for a, r in removed}
    uncovered = [
        Uncovered(start_s=s.start_s, end_s=s.end_s, category=taxonomy.category_of(leaf),
                  leaf=leaf, note=why.get((s.start_s, leaf), ""))
        for s in inventory.segments for leaf in s.leaves if (s.start_s, leaf) not in assessed
    ]

    cited_ids = sorted({pid for f in findings for pid in f.rule_card_ids})
    cited: list[Provision] = [by_id[pid] for pid in cited_ids]
    laws = provisions.law_infos(country)
    if v.removed_citations:
        notes = notes + [f"Verifier removed {v.removed_citations} citation(s) that did not fit."]
    return CountryReport(
        country=country,
        law="; ".join(law.title for law in laws if law.provisions) or config.COUNTRIES[country].law,
        label=None,
        overall_risk=pipeline.overall_risk(findings, checked=bool(provisions.provisions_for(country))),
        findings=findings,
        rules_status="reviewed" if all(p.review.status == "reviewed" for p in provisions.provisions_for(country))
        else "draft/unreviewed",
        risk_framing=_framing(findings),
        cited_cards=[provisions.to_card(p) for p in cited],
        dropped_findings=v.dropped,
        uncovered=uncovered,
        brain="adk",
        laws=laws,
        cited_provisions=cited,
        platform_obligations=platform_duties,
        critic_changes=critic_changes,
        notes=notes,
    )


async def assess_country(inventory: Inventory, country: str, *, mock: bool) -> CountryReport:
    """assessor -> verifier -> critic -> report, for one country (no Workflow; used by tests)."""
    assessments, notes = await agents.assess(inventory, country, mock=mock)
    v = verifier.verify(assessments, country, inventory)
    verifier_event(country, v)
    verdicts, c_notes = await agents.critique(v.covered, inventory, country, mock=mock)
    kept, removed, changes = agents.apply_verdicts(v.covered, verdicts)
    agents.critic_event(country, v.covered, verdicts, removed, changes)
    return build_report(country, inventory, kept, removed, v, changes, notes + c_notes)


def verifier_event(country: str, v: verifier.Verified) -> None:
    """The citation check (plain code): what it let through and what it removed."""
    if not (v.dropped or v.removed_citations):
        if v.covered:
            events.emit("review", f"Citation check passed: {len(v.covered)} cited finding(s), every "
                        "provision id exists and targets the moment", stage="country", country=country)
        return
    parts = []
    if v.dropped:
        parts.append(f"dropped {v.dropped} assessment(s) whose citations did not exist or fit")
    if v.removed_citations:
        parts.append(f"removed {v.removed_citations} citation(s) that did not fit")
    events.emit("review", "Citation check " + " and ".join(parts), stage="country", country=country)


def _legacy_country(inventory: Inventory, country: str, mock: bool) -> CountryReport:
    cards, status = pipeline.load_cards(country)
    rep = pipeline.stage_b(inventory, country, cards, rules_status=status, mock=mock)
    if cards:
        rep.notes.append("No provisions loaded for this country yet: checked against legacy rule cards.")
    return rep


async def run(
    video_path: Path, duration_s: float | None, countries: list[str], *, mock: bool,
    on_stage: StageHook | None = None,
    on_start: Callable[[str], Awaitable[None]] | None = None,
    on_done: Callable[[str, CountryReport | None, str | None], Awaitable[None]] | None = None,
    on_inventory: Callable[[Inventory], Awaitable[None]] | None = None,
) -> tuple[Inventory, dict[str, CountryReport | Exception]]:
    """Run the brain Workflow. Per-country failures are returned, not raised."""
    from google.adk import Workflow
    from google.adk.workflow import START, FunctionNode, JoinNode

    state: dict[str, Any] = {"inventory": None, "results": {}}

    async def perceiver(node_input: Any) -> dict:
        if on_stage:
            await on_stage("inventory", "running", "")
        if mock:
            await asyncio.sleep(config.mock_delay_s())
        with metrics.stage("stage_a"):
            inv = await asyncio.to_thread(pipeline.stage_a, video_path, duration_s, mock=mock)
        state["inventory"] = inv
        tagged = sum(1 for s in inv.segments if s.leaves)
        if on_stage:
            await on_stage("inventory", "done", f"{len(inv.segments)} segments, {tagged} sensitive")
        if on_inventory:
            await on_inventory(inv)
        return {"segments": len(inv.segments)}

    def chain(cc: str) -> tuple[FunctionNode, FunctionNode, FunctionNode]:
        ctx: dict[str, Any] = {}

        async def fail(e: Exception) -> dict:
            log.error("brain failed for %s: %s", cc, type(e).__name__)
            state["results"][cc] = e
            if on_done:
                await on_done(cc, None, pipeline.public_error(e))
            return {"country": cc, "error": True}

        async def assessor(node_input: Any) -> dict:
            if on_start:
                await on_start(cc)
            inv = state["inventory"]
            try:
                with metrics.stage(f"assessor:{cc}"):
                    if mock:
                        await asyncio.sleep(config.mock_delay_s())
                    if not provisions.has_provisions(cc):
                        ctx["report"] = await asyncio.to_thread(_legacy_country, inv, cc, mock)
                        return {"country": cc, "legacy": True}
                    if on_stage:
                        n = sum(1 for s in inv.segments if s.leaves)
                        await on_stage(f"country:{cc}", "running",
                                       f"assess: {n} sensitive moment(s)")
                    ctx["assessments"], ctx["notes"] = await agents.assess(inv, cc, mock=mock)
                return {"country": cc, "assessments": len(ctx["assessments"])}
            except Exception as e:  # noqa: BLE001 - reported per country
                return await fail(e)

        async def verify(node_input: dict) -> dict:
            if node_input.get("error") or "report" in ctx:
                return node_input
            with metrics.stage(f"verifier:{cc}"):
                ctx["verified"] = v = verifier.verify(ctx["assessments"], cc, state["inventory"])
            verifier_event(cc, v)
            return {"country": cc, "covered": len(v.covered), "dropped": v.dropped}

        async def critic(node_input: dict) -> dict:
            if node_input.get("error"):
                return node_input
            try:
                if "report" not in ctx:
                    v = ctx["verified"]
                    if on_stage and v.covered:
                        await on_stage(f"country:{cc}", "running",
                                       f"critic: re-checking {len(v.covered)} finding(s)")
                    with metrics.stage(f"critic:{cc}"):
                        verdicts, notes = await agents.critique(v.covered, state["inventory"], cc,
                                                                mock=mock)
                    kept, removed, changes = agents.apply_verdicts(v.covered, verdicts)
                    agents.critic_event(cc, v.covered, verdicts, removed, changes)
                    ctx["report"] = build_report(cc, state["inventory"], kept, removed, v, changes,
                                                 ctx["notes"] + notes)
            except Exception as e:  # noqa: BLE001
                return await fail(e)
            rep: CountryReport = ctx["report"]
            state["results"][cc] = rep
            if on_done:
                await on_done(cc, rep, None)
            return {"country": cc, "risk": rep.overall_risk, "findings": len(rep.findings)}

        return (FunctionNode(func=assessor, name=f"assessor_{cc}"),
                FunctionNode(func=verify, name=f"verifier_{cc}"),
                FunctionNode(func=critic, name=f"critic_{cc}"))

    def reporter(node_input: Any) -> dict:
        return {"countries": sorted(state["results"])}

    chains = [chain(cc) for cc in countries]
    perceive_node = FunctionNode(func=perceiver, name="perceiver")
    join = JoinNode(name="join")
    edges: list = [(START, perceive_node, tuple(c[0] for c in chains))]
    edges += [(a, v, c, join) for a, v, c in chains]
    edges.append((join, FunctionNode(func=reporter, name="reporter")))
    wf = Workflow(name="brain", edges=edges, max_concurrency=max(1, config.GEMINI_CONCURRENCY))

    await _run_workflow(wf)
    if state["inventory"] is None:
        raise RuntimeError("perceiver did not produce an inventory")
    return state["inventory"], {cc: state["results"].get(cc, RuntimeError("no result"))
                                for cc in countries}


async def _run_workflow(wf) -> None:
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(node=wf, app_name="streamclearance-brain")
    session = await runner.session_service.create_session(app_name="streamclearance-brain",
                                                          user_id="job")
    async for ev in runner.run_async(
            user_id="job", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="analyse")])):
        if ev.error_message:
            raise RuntimeError(f"brain workflow: {ev.error_message}")
