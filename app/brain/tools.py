"""Tools the country assessor agent can call (Layer 2 recall, exposed to the model).

Each assessor is bound to one country: a call for another country is refused,
so an agent can never cite a foreign provision.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from app import taxonomy, config, events, metrics, retrieval
from app.schemas import Provision


def provision_view(p: Provision) -> dict[str, Any]:
    """What the model sees of a provision (full verbatim text + exceptions)."""
    return {
        "id": p.id, "law": p.law_title, "law_status": p.law_status, "section": p.section,
        "page": p.page, "targets": p.targets, "summary": p.summary,
        "verbatim_text": p.verbatim_text, "exceptions": p.exceptions,
        "duty_holder": p.duty_holder,
    }


LIMIT_REACHED = "tool-call limit reached: answer now with the provisions you already have"


# Readable short names for the feed (internal law ids stay in the report JSON).
SHORT_LAW = {
    "OSA2025": "Online Safety Act 2025", "MAS1956": "Medicines Act 1956",
    "MOHADV": "MOH advertising guideline", "CGHA1953": "Common Gaming Houses Act",
    "BETTING1953": "Betting Act 1953", "ITRULES2021": "IT Rules 2021",
    "PROGA2025": "Online Gaming Act 2025", "DMR1954": "Drugs and Magic Remedies Act",
    "BNS2023": "Bharatiya Nyaya Sanhita", "OSA2021": "Online Safety Act 2021",
    "IGA2001": "Interactive Gambling Act", "TGACODE2021": "TGA Advertising Code",
    "OSRA2025": "Online Safety (Relief) Act", "GCA2022": "Gambling Control Act",
    "PECA2025": "PECA 2025",
}


def law_name(law_id: str) -> str:
    return SHORT_LAW.get(law_id, law_id)


def cite(p: Provision) -> str:
    """Short reference for the feed, e.g. "TGA Advertising Code s9(1), p.13"."""
    section = p.section if len(p.section) <= 32 else p.section.split("(")[0].strip()[:32] + "…"
    return f"{law_name(p.law_id)} {section}, p.{p.page}"


def found(hits: list[Provision], n_passages: int = 0) -> str:
    parts = [cite(p) for p in hits[:4]] + ([f"+{len(hits) - 4} more"] if len(hits) > 4 else [])
    if n_passages:
        parts.append(f"{n_passages} passage{'s' * (n_passages != 1)} of law text")
    return "; ".join(parts) or "no matching provision"


def make_tools(bound: str, max_calls: int | None = None
               ) -> tuple[Callable[..., dict], Callable[..., dict]]:
    """Tools bound to one country; together they answer at most `max_calls` calls
    (None: no limit), then return an error telling the agent to answer."""
    used = [0]

    def over_limit() -> bool:
        used[0] += 1
        return max_calls is not None and used[0] > max_calls

    def lookup_provisions(country: str, leaves: list[str]) -> dict:
        """Provisions of a country's loaded laws that target the given content-taxonomy leaves.

        Args:
          country: two-letter country code, e.g. "MY".
          leaves: taxonomy ids of the moment, e.g. ["health.cure_claim"].
        """
        if country.strip().upper() != bound:
            return {"error": f"you assess {bound} only", "provisions": []}
        if over_limit():
            return {"error": LIMIT_REACHED, "provisions": []}
        with metrics.timed(f"retrieval:{bound}"):
            hits = retrieval.lookup(bound, leaves)
        events.emit("search", f"Looked up {config.COUNTRIES[bound].name} law for "
                    f"{', '.join(taxonomy.label(str(x)).lower() for x in leaves)} → {found(hits)}", stage="country", country=bound)
        return {"provisions": [provision_view(p) for p in hits]}

    async def search_law(country: str, query: str) -> dict:
        """Free-text search over a country's loaded laws (use when lookup finds nothing).

        Returns matching provisions, plus passages of the full law text (law, page,
        snippet) that no provision covers: passages are context only; cite provision ids.

        Args:
          country: two-letter country code, e.g. "MY".
          query: words describing the moment, e.g. "herbal drink cures diabetes".
        """
        if country.strip().upper() != bound:
            return {"error": f"you assess {bound} only", "provisions": []}
        if over_limit():
            return {"error": LIMIT_REACHED, "provisions": []}
        # a network query (RAG Engine): keep it off the event loop the other countries share
        with metrics.timed(f"retrieval:{bound}"):
            hits, passages = await asyncio.to_thread(retrieval.search_with_passages, bound, query, 5)
        events.emit("search", f"Searched {config.COUNTRIES[bound].name} law for {events.quote(query, 60)} "
                    f"→ {found(hits, len(passages))}", stage="country", country=bound)
        return {"provisions": [provision_view(p) for p in hits],
                "passages": [{"law": x.law, "page": x.page, "snippet": x.snippet} for x in passages]}

    return lookup_provisions, search_law
