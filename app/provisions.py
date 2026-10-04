"""Provision store (Layer 1 of the brain) and the law registry.

Laws are listed in data/laws/registry.json (a country can have several).
Provisions live in data/provisions/<cc>/<law_id>.json, drafted by
tools/build_provisions.py and reviewed by people (git is the audit trail).
They are loaded once into memory; app/retrieval.py indexes them.

Rule cards (app.schemas.RuleCard) are a derived view: `cards_from_provisions`
lets the legacy stage B and the old UI keep working on provision data.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from app import config, taxonomy
from app.schemas import LawInfo, LawStatus, Provision, RuleCard

log = logging.getLogger(__name__)

REGISTRY_PATH = config.DATA_DIR / "laws" / "registry.json"
PROVISIONS_DIR = config.DATA_DIR / "provisions"


@dataclass(frozen=True)
class Law:
    id: str  # unique within its country, e.g. "OSA2025"
    country: str
    title: str
    status: LawStatus
    pdf: str  # file name in data/laws/raw
    text: str  # file name in data/laws/text
    label: str | None = None

    @property
    def key(self) -> str:
        return f"{self.country}/{self.id}"

    @property
    def pdf_path(self) -> Path:
        return config.LAWS_RAW_DIR / self.pdf

    @property
    def text_path(self) -> Path:
        return config.LAWS_TEXT_DIR / self.text

    @property
    def provisions_path(self) -> Path:
        return PROVISIONS_DIR / self.country.lower() / f"{self.id}.json"

    @property
    def pdf_url(self) -> str:
        return f"/api/laws/{self.country}/{self.id}/pdf"


def registry() -> list[Law]:
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [Law(**item) for item in raw]


def laws_for(country: str) -> list[Law]:
    return [law for law in registry() if law.country == country]


def get_law(key: str) -> Law:
    """Look up by "MY/OSA2025", or by bare id when it is unique."""
    laws = registry()
    hits = [law for law in laws if law.key == key] or [law for law in laws if law.id == key]
    if len(hits) != 1:
        raise KeyError(f"unknown or ambiguous law {key!r}")
    return hits[0]


# ---------------------------------------------------------------- store

_lock = threading.Lock()
_cache: dict[str, list[Provision]] | None = None


def _load_all() -> dict[str, list[Provision]]:
    out: dict[str, list[Provision]] = {}
    for law in registry():
        path = law.provisions_path
        if not path.exists():
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        provs = [Provision.model_validate(p) for p in items]
        bad = [p.id for p in provs if p.country != law.country or p.law_id != law.id]
        if bad:
            raise ValueError(f"{path}: provisions belong to another law: {bad[:3]}")
        out.setdefault(law.country, []).extend(provs)
    return out


def all_provisions() -> dict[str, list[Provision]]:
    global _cache
    with _lock:
        if _cache is None:
            _cache = _load_all()
            log.info("provisions loaded: %s", {cc: len(v) for cc, v in _cache.items()})
        return _cache


def reload() -> None:
    global _cache
    with _lock:
        _cache = None
    from app import retrieval

    retrieval.reset()


def provisions_for(country: str) -> list[Provision]:
    return list(all_provisions().get(country, []))


def by_id(country: str) -> dict[str, Provision]:
    return {p.id: p for p in provisions_for(country)}


def has_provisions(country: str) -> bool:
    return bool(all_provisions().get(country))


def law_infos(country: str) -> list[LawInfo]:
    counts: dict[str, int] = {}
    for p in provisions_for(country):
        counts[p.law_id] = counts.get(p.law_id, 0) + 1
    return [LawInfo(id=law.id, title=law.title, status=law.status, label=law.label,
                    pdf_url=law.pdf_url, provisions=counts.get(law.id, 0))
            for law in laws_for(country)]


# ---------------------------------------------------------------- derived rule cards

_APPLIES_TO = {"platform": "platform", "creator": "creator", "advertiser": "creator", "both": "both"}


def to_card(p: Provision) -> RuleCard:
    from tools.build_rules import QUOTE_MAX, shorten_quote

    c = p.consequence
    return RuleCard(
        id=p.id, country=p.country, law=p.law_title, section=p.section, page=p.page,
        category=taxonomy.category_of(p.targets[0]),
        conduct=p.summary or ", ".join(taxonomy.label(t) for t in p.targets),
        applies_to=_APPLIES_TO[p.duty_holder],
        platform_duty=f"{c.kind}; {c.timeframe}",
        penalty=c.penalty,
        quote=shorten_quote(p.verbatim_text, QUOTE_MAX),
        needs_review=p.review.status != "reviewed",
    )


def cards_from_provisions(country: str) -> list[RuleCard]:
    return [to_card(p) for p in provisions_for(country)]
