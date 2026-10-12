"""Severity floors: the same video gets the same rating, whatever the model's mood.

Applied in code after the critic (app/pipeline.py `finalize`). A floor only RAISES a
finding's risk, never lowers it, and only when:
- the finding's leaf is in the table below and no exception was found to apply;
- at least one cited provision genuinely matches: a content provision of an in-force
  law (not a bill, not a guideline) that targets the leaf and binds the person who
  publishes (creator, advertiser or both), so the conduct is an offence for them;
- the moment's own words meet the floor's test (e.g. a serious disease is named).

| leaf | floor | moment test |
|---|---|---|
| gambling.promotion | red | none: promoting gambling where the cited law makes that an offence |
| health.cure_claim | red | names a serious disease, or tells viewers to stop their medicine |
| health.misinformation | red | tells viewers to stop their medicine or treatment |
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from app import events, provisions, taxonomy
from app.schemas import RISK_ORDER, CountryReport, Finding, Inventory, Risk

# Diseases the medicine-advertising laws single out as serious (e.g. MY Medicine
# Advertisements and Sale Act 1956, Schedule; TGA Code "serious forms of disease").
SERIOUS_DISEASE = re.compile(
    r"\b(diabet\w*|cancer\w*|tumou?rs?|leuka?emia|hiv|aids|heart (?:disease|attack|failure)|strokes?|"
    r"kidney (?:disease|failure)|hypertension|high blood pressure|epilep\w*|tuberculosis|asthma|"
    r"hepatitis|cirrhosis|alzheimer\w*|parkinson\w*|dementia|paraly\w*|schizophreni\w*|covid\w*|"
    r"sexually transmitted|infertil\w*|blindness|deafness)\b", re.IGNORECASE)
STOP_MEDICINE = re.compile(
    r"\b(stop|stopped|quit|throw away|ditch|give up|gave up|no need for|don'?t need|no more|skip|"
    r"instead of)\b[\w\s',]{0,30}?\b(medicin\w*|medication\w*|insulin|pills?|tablets?|treatment|"
    r"prescription\w*|chemo\w*|metformin|doctor'?s? drugs?)\b", re.IGNORECASE)


def _serious_or_stop(text: str) -> bool:
    return bool(SERIOUS_DISEASE.search(text) or STOP_MEDICINE.search(text))


def _stop(text: str) -> bool:
    return bool(STOP_MEDICINE.search(text))


@dataclass(frozen=True)
class Floor:
    leaf: str
    risk: Risk
    why: str
    test: Callable[[str], bool] | None = None
    duty_holders: tuple[str, ...] = ("creator", "advertiser", "both")
    law_statuses: tuple[str, ...] = ("in_force", "amended")


FLOORS: dict[str, Floor] = {f.leaf: f for f in [
    Floor("gambling.promotion", "red",
          "promotes gambling, and a cited in-force provision makes that an offence for the publisher"),
    Floor("health.cure_claim", "red",
          "claims to cure a serious disease or tells viewers to stop their medicine, under a cited "
          "in-force provision that binds the advertiser", _serious_or_stop),
    Floor("health.misinformation", "red",
          "tells viewers to stop their medicine or treatment, under a cited in-force provision that "
          "binds the publisher", _stop),
]}


def moment_text(inventory: Inventory, f: Finding) -> str:
    """What the moment says and shows in words (transcript, claims, on-screen text)."""
    return " ".join(
        " ".join([s.transcript, *s.claims, s.on_screen_text])
        for s in inventory.segments
        if (s.start_s < f.end_s and f.start_s < s.end_s) or s.start_s == f.start_s)


def matching_provisions(country: str, f: Finding, floor: Floor) -> list[str]:
    by_id = provisions.by_id(country)
    out = []
    for pid in f.rule_card_ids:
        p = by_id.get(pid)
        if p and p.kind == "content" and p.law_status in floor.law_statuses \
                and p.duty_holder in floor.duty_holders \
                and any(taxonomy.covers(t, floor.leaf) for t in p.targets):
            out.append(pid)
    return out


def apply(rep: CountryReport, inventory: Inventory) -> int:
    """Raise findings to their floor (in place). Returns how many were raised."""
    raised = 0
    for f in rep.findings:
        floor = FLOORS.get(f.leaf or "")
        if floor is None or f.video_level or f.exception_applies \
                or RISK_ORDER[f.risk] >= RISK_ORDER[floor.risk]:
            continue
        if floor.test is not None and not floor.test(moment_text(inventory, f)):
            continue
        ids = matching_provisions(rep.country, f, floor)
        if not ids:
            continue
        old, f.risk = f.risk, floor.risk
        f.floor_note = (f"Raised from {old} to {floor.risk} by a fixed rule: the moment {floor.why} "
                        f"({', '.join(ids)}).")
        raised += 1
        events.emit("review", f"Fixed severity rule raised {events.span(f.start_s, f.end_s)} "
                    f"{taxonomy.label(floor.leaf)} from {old} to {floor.risk} ({', '.join(ids)})",
                    stage="country", country=rep.country, ref=events.span(f.start_s, f.end_s))
    if raised and rep.overall_risk != "not_checked":
        worst = max((f.risk for f in rep.findings), key=RISK_ORDER.__getitem__)
        if RISK_ORDER[worst] > RISK_ORDER[rep.overall_risk]:
            rep.overall_risk = worst
    return raised
