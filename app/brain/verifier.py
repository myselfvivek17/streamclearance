"""Verifier (plain Python, not an LLM): the brain's anti-hallucination gate.

Per assessment that claims coverage:
- timestamps must lie inside the video;
- the leaf must be a taxonomy leaf that stage A gave to an overlapping moment;
- every cited provision id must exist and belong to this country;
- the moment's leaf must be covered by the provision's targets (leaf or parent);
- the provision must be a content provision (platform obligations are attached
  by the report, never cited as "covered by").
Citations failing the last two are removed; an assessment left with none is
dropped and counted. Assessments with no citation are "not covered".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app import provisions, taxonomy
from app.brain.agents import MomentAssessment
from app.schemas import Inventory

log = logging.getLogger(__name__)


@dataclass
class Verified:
    covered: list[MomentAssessment] = field(default_factory=list)
    not_covered: list[MomentAssessment] = field(default_factory=list)
    dropped: int = 0  # assessments dropped entirely
    removed_citations: int = 0  # bad ids removed from assessments that were kept
    reasons: list[str] = field(default_factory=list)


def _leaves_at(inventory: Inventory, start: float, end: float) -> set[str]:
    return {leaf for s in inventory.segments
            if (s.start_s < end and start < s.end_s) or s.start_s == start
            for leaf in s.leaves}


def verify(assessments: list[MomentAssessment], country: str, inventory: Inventory) -> Verified:
    out = Verified()
    known = provisions.by_id(country)
    seen: set[tuple[float, str]] = set()

    def drop(a: MomentAssessment, why: str) -> None:
        log.warning("%s: verifier dropped %.1f-%.1fs %s: %s", country, a.start_s, a.end_s, a.leaf, why)
        out.dropped += 1
        out.reasons.append(why)

    for a in assessments:
        leaf = a.leaf.strip().lower()
        if not (0 <= a.start_s <= a.end_s <= inventory.duration_s + 1e-6):
            drop(a, f"timestamps {a.start_s}-{a.end_s} outside 0-{inventory.duration_s}")
            continue
        if not taxonomy.is_leaf(leaf) or leaf not in _leaves_at(inventory, a.start_s, a.end_s):
            drop(a, f"leaf {a.leaf!r} was not tagged on this moment")
            continue
        a = a.model_copy(update={"leaf": leaf})
        if not a.covered_by:
            out.not_covered.append(a)
            continue
        good = [pid for pid in a.covered_by
                if pid in known and known[pid].kind == "content"
                and any(taxonomy.covers(t, leaf) for t in known[pid].targets)]
        bad = [pid for pid in a.covered_by if pid not in good]
        if not good:
            drop(a, f"citations {bad} unknown, foreign or not targeting {leaf}")
            continue
        if (a.start_s, leaf) in seen:
            continue  # duplicate (moment, leaf): first verified one wins
        seen.add((a.start_s, leaf))
        out.removed_citations += len(bad)
        out.covered.append(a.model_copy(update={"covered_by": list(dict.fromkeys(good))}))
    nc: dict[tuple[float, str], MomentAssessment] = {}
    for a in out.not_covered:
        if (a.start_s, a.leaf) not in seen:
            nc.setdefault((a.start_s, a.leaf), a)
    out.not_covered = list(nc.values())
    return out
