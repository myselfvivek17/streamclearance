"""Retrieval (Layer 2 of the brain): how the brain recalls provisions.

Path A, `lookup(country, leaves)`: exact (country, taxonomy leaf) index.
Deterministic and explainable; the primary path.

Path B, `search(country, query, k)`: free-text search. Today a local BM25
over provision text (pure Python). TODO(file-search): when GEMINI_API_KEY
exists, switch to a Gemini File Search store (one store, metadata `country`,
`law`, `status`; filter on `country`) via `_file_search`, keeping BM25 as the
offline/mock fallback.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter

from app import config, provisions, taxonomy
from app.schemas import Provision, ProvisionKind

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


def _file_search(country: str, query: str, k: int) -> list[Provision] | None:
    """TODO(file-search): Gemini File Search, metadata filter `country = <cc>`.

    Return provisions mapped back from the store's chunk metadata (`provision_id`),
    or None to fall back to BM25. Not wired yet: there is no Gemini key.
    """
    return None


def search(country: str, query: str, k: int = 5) -> list[Provision]:
    """Free-text search within one country's content provisions."""
    k = max(1, min(k, MAX_RESULTS))
    if config.provider() == "gemini":
        hits = _file_search(country, query, k)
        if hits is not None:
            return hits
    return [p for _, p in _get_bm25(country).search(query, k)]
