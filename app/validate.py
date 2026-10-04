"""Post-validation of model outputs (anti-hallucination checks)."""
from __future__ import annotations

import logging
from collections.abc import Iterable

from app.schemas import Finding, RuleCard

log = logging.getLogger(__name__)


def validate_findings(
    findings: Iterable[Finding], cards: Iterable[RuleCard], duration_s: float
) -> list[Finding]:
    """Drop findings that cite unknown rule-card IDs or fall outside the video."""
    known = {c.id for c in cards}
    kept: list[Finding] = []
    for f in findings:
        unknown = [cid for cid in f.rule_card_ids if cid not in known]
        if unknown:
            log.warning("dropping finding at %.1fs: unknown card ids %s", f.start_s, unknown)
            continue
        if f.start_s < 0 or f.end_s > duration_s or f.start_s > f.end_s:
            log.warning(
                "dropping finding: timestamps %.1f-%.1f outside 0-%.1f",
                f.start_s, f.end_s, duration_s,
            )
            continue
        kept.append(f)
    return kept
