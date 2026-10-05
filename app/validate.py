"""Post-validation of model outputs (anti-hallucination checks)."""
from __future__ import annotations

import logging
from collections.abc import Iterable

from app.schemas import Finding, RuleCard

log = logging.getLogger(__name__)


def validate_findings(
    findings: Iterable[Finding], cards: Iterable[RuleCard], duration_s: float
) -> list[Finding]:
    """Drop findings that cite unknown rule-card IDs, fall outside the video, or whose
    category is not targeted by any cited card (cards of category "other" are general
    procedure and can support a finding but never justify one on their own)."""
    by_id = {c.id: c for c in cards}
    known = set(by_id)
    kept: list[Finding] = []
    for f in findings:
        unknown = [cid for cid in f.rule_card_ids if cid not in known]
        if unknown:
            log.warning("dropping finding at %.1fs: %d unknown card ids", f.start_s, len(unknown))
            continue
        if f.start_s < 0 or f.end_s > duration_s or f.start_s > f.end_s:
            log.warning(
                "dropping finding: timestamps %.1f-%.1f outside 0-%.1f",
                f.start_s, f.end_s, duration_s,
            )
            continue
        if not any(by_id[cid].category == f.category for cid in f.rule_card_ids):
            log.warning(
                "dropping finding at %.1fs: category %s not targeted by cited cards %s",
                f.start_s, f.category.value, f.rule_card_ids,
            )
            continue
        kept.append(f)
    return kept
