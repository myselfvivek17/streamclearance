"""Retrieval (Layer 2 of the brain): how the brain recalls provisions.

Path A, `lookup(country, leaves)`: exact (country, taxonomy leaf) index.
Deterministic and explainable; the primary path.

Path B, `search(country, query, k)`: free-text search. On the Vertex backend
it queries the Google document search (Vertex AI RAG Engine, app/law_search.py:
one corpus per country holding every provision and every page of every law),
mapping hits back to provisions; full-text hits map to the best provision on
that law page, or come back as passages (law, page, snippet). Local BM25 over
provision text is the offline/mock fallback, and the fallback on any search
error (logged, and noted in the country report via `fallback_notes`).
"""
from __future__ import annotations

import contextvars
import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass

from app import config, law_search, provisions, taxonomy
from app.schemas import Provision, ProvisionKind

log = logging.getLogger(__name__)

MAX_RESULTS = 10
_STATUS_RANK = {"in_force": 0, "amended": 1, "guideline": 2, "bill": 3}
_REVIEW_RANK = {"reviewed": 0, "draft": 1}
# a rule binding the person who posts (creator/advertiser) is more on point than a
# general list that also binds platforms ("both"), or a platform-only rule
_DUTY_RANK = {"creator": 0, "advertiser": 0, "both": 1, "platform": 2}


def rank_key(p: Provision) -> tuple[int, int]:
    """reviewed > draft, then in_force > amended > guideline > bill."""
    return _REVIEW_RANK[p.review.status], _STATUS_RANK[p.law_status]


# ---------------------------------------------------------------- path A: exact lookup

_lock = threading.Lock()
_index: dict[tuple[str, str], list[Provision]] | None = None


def _build_index() -> dict[tuple[str, str], list[Provision]]:
    idx: dict[tuple[str, str], list[Provision]] = {}
    for cc, provs in provisions.all_provisions().items():
        for p in provs:
            for t in p.targets:
                idx.setdefault((cc, t), []).append(p)
    return idx


def _get_index() -> dict[tuple[str, str], list[Provision]]:
    global _index
    with _lock:
        if _index is None:
            _index = _build_index()
        return _index


def _content_only(provs: list[Provision]) -> list[Provision]:
    return [p for p in provs if p.kind == "content"]


def reset() -> None:
    global _index, _bm25
    with _lock:
        _index, _bm25 = None, {}


def lookup(country: str, leaves: list[str], *, cap: int = MAX_RESULTS,
           kind: ProvisionKind = "content") -> list[Provision]:
    """Provisions of `country` (of `kind`) targeting any of `leaves`.

    Exact leaf matches first, then provisions that target the leaf's parent
    (e.g. a law on all of "gambling"); a bare parent in `leaves` matches every
    provision under it. Within each tier: reviewed > draft, in_force > guideline
    > bill, then the most specific (fewest targets), then rules binding the
    creator/advertiser before those also or only binding platforms. At most `cap`
    results, no duplicates.
    """
    idx = _get_index()
    exact: list[Provision] = []
    parent: list[Provision] = []
    for value in leaves:
        value = value.strip().lower()
        if taxonomy.is_leaf(value):
            exact += idx.get((country, value), [])
            parent += idx.get((country, taxonomy.parent_of(value)), [])
        elif value in taxonomy.parents():
            exact += idx.get((country, value), [])
            for child in taxonomy.children(value):
                parent += idx.get((country, child), [])
    out: list[Provision] = []
    seen: set[str] = set()
    for tier in (exact, parent):
        for p in sorted(tier, key=lambda p: (*rank_key(p), len(p.targets), _DUTY_RANK[p.duty_holder])):
            if p.kind == kind and p.id not in seen:
                seen.add(p.id)
                out.append(p)
    return out[:cap]


def obligations(country: str, leaves: list[str], *, cap: int = 6) -> list[Provision]:
    """Platform-obligation provisions (duties, takedown, penalties) for the leaves found.

    Ranked: reviewed/in-force first, then the most specific duties (fewest targets,
    e.g. priority-content takedown before general duties), then those stating a
    penalty or timeframe; at most `cap`.
    """
    def stated(p: Provision) -> int:
        c = p.consequence
        return 0 if any(not v.startswith("not specified") for v in (c.penalty, c.timeframe)) else 1

    hits = lookup(country, leaves, cap=10_000, kind="obligation")
    hits.sort(key=lambda p: (*rank_key(p), len(p.targets), stated(p)))
    return hits[:cap]


# ---------------------------------------------------------------- path B: search

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("""a an and are as at be by for from has have in is it its may of on or that the
this to was which with any not shall such other""".split())


def tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP or len(t) < 2:
            continue
        # tiny stemmer: cures/cured/curing -> cure, diseases -> disease
        for suf in ("ing", "es", "ed", "s"):
            if len(t) > 4 and t.endswith(suf):
                t = t[: -len(suf)]
                break
        out.append(t)
    return out


def _doc_text(p: Provision) -> str:
    labels = " ".join(taxonomy.label(t) + " " + t.replace(".", " ").replace("_", " ") for t in p.targets)
    return " ".join([p.law_title, p.section, p.summary, p.verbatim_text, labels, *p.exceptions])


class BM25:
    def __init__(self, docs: list[Provision], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(_doc_text(d))) for d in docs]
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[float, Provision]]:
        q = tokens(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                s += self.idf[t] * f * (self.k1 + 1) / norm
            if s > 0:
                scored.append((s, self.docs[i]))
        scored.sort(key=lambda x: (-x[0], rank_key(x[1])))
        return scored[:k]


_bm25: dict[str, BM25] = {}


def _get_bm25(country: str) -> BM25:
    with _lock:
        if country not in _bm25:
            _bm25[country] = BM25(_content_only(provisions.provisions_for(country)))
        return _bm25[country]


@dataclass(frozen=True)
class Passage:
    """A full-text hit that no provision on that page matches: context, not citable."""
    law_id: str
    law: str
    page: int
    snippet: str


SNIPPET_CHARS = 600
MAX_PASSAGES = 3

_fallbacks: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "search_fallbacks", default=None)


def track_fallbacks() -> list[str]:
    """Collect search-fallback notes in the current context (one assessor run)."""
    notes: list[str] = []
    _fallbacks.set(notes)
    return notes


def _note_fallback(country: str, e: Exception) -> None:
    log.warning("%s: law search (RAG Engine) failed (%s); using local BM25", country, type(e).__name__)
    notes = _fallbacks.get()
    msg = (f"Law search fallback: Vertex AI RAG Engine query failed ({type(e).__name__}); "
           "free-text search used local keyword matching (BM25) instead.")
    if notes is not None and msg not in notes:
        notes.append(msg)


def _overlap(a: str, b: str) -> int:
    return len(set(tokens(a)) & set(tokens(b)))


def _file_search(country: str, query: str, k: int, kind: ProvisionKind | None = "content",
                 ) -> tuple[list[Provision], list[Passage]] | None:
    """Vertex AI RAG Engine query on the country's corpus, mapped back to provisions.

    None when the country has no corpus in data/search_index.json (use BM25).
    Raises on service errors (the caller falls back).
    """
    index = law_search.load_index()
    corpus = index.get("corpora", {}).get(country)
    if not corpus:
        return None
    docs = index.get("docs", {})
    by_id = provisions.by_id(country)
    by_page: dict[tuple[str, int], list[Provision]] = {}
    for p in by_id.values():
        by_page.setdefault((p.law_id, p.page), []).append(p)
    titles = {law.id: law.title for law in provisions.laws_for(country)}

    hits: list[Provision] = []
    passages: list[Passage] = []
    for c in law_search.retrieve(corpus, query, top_k=min(3 * k + 5, 30),
                                 location=index.get("location")):
        meta = docs.get(c.doc)
        if not meta or meta.get("country") != country:
            continue
        if meta.get("doc_type") == "provision":
            p = by_id.get(meta.get("provision_id", ""))
            if p and (kind is None or p.kind == kind) and p not in hits:
                hits.append(p)
            continue
        page = int(meta.get("page", 0))
        body = c.text.split("\n---\n", 1)[-1]
        on_page = [p for p in by_page.get((meta.get("law_id", ""), page), [])
                   if kind is None or p.kind == kind]
        if on_page:
            best = max(on_page, key=lambda p: (_overlap(_doc_text(p), body), -rank_key(p)[0]))
            if best not in hits:
                hits.append(best)
        elif len(passages) < MAX_PASSAGES:
            law_id = meta.get("law_id", "")
            title = titles.get(law_id, law_id)
            body = body.strip().removeprefix(f"{title}, page {page}")  # the doc's own heading
            passages.append(Passage(law_id=law_id, law=title, page=page,
                                    snippet=" ".join(body.split())[:SNIPPET_CHARS]))
    # The query holds words from the video: log its length, not its text.
    log.info("law search %s (%d chars): %d provisions, %d passages (RAG Engine)",
             country, len(query), len(hits[:k]), len(passages))
    return hits[:k], passages


def search_with_passages(country: str, query: str, k: int = 5,
                         kind: ProvisionKind | None = "content") -> tuple[list[Provision], list[Passage]]:
    """Free-text search within one country: (provisions, full-text passages).

    `kind` = "content" (default), "obligation", or None for both. Passages come only
    from the document search; BM25 searches content provisions only.
    """
    k = max(1, min(k, MAX_RESULTS))
    if config.law_search() == "rag":
        try:
            found = _file_search(country, query, k, kind)
            if found is not None:
                return found
        except Exception as e:  # noqa: BLE001 - any search error: fall back, say so
            _note_fallback(country, e)
    if kind == "obligation":
        return [], []
    return [p for _, p in _get_bm25(country).search(query, k)], []


def search(country: str, query: str, k: int = 5, kind: ProvisionKind | None = "content") -> list[Provision]:
    """Free-text search within one country's provisions (content by default)."""
    return search_with_passages(country, query, k, kind)[0]
