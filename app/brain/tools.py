"""Tools the country assessor agent can call (Layer 2 recall, exposed to the model).

Each assessor is bound to one country: a call for another country is refused,
so an agent can never cite a foreign provision.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from app import retrieval
from app.schemas import Provision


def provision_view(p: Provision) -> dict[str, Any]:
    """What the model sees of a provision (full verbatim text + exceptions)."""
    return {
        "id": p.id, "law": p.law_title, "law_status": p.law_status, "section": p.section,
        "page": p.page, "targets": p.targets, "summary": p.summary,
        "verbatim_text": p.verbatim_text, "exceptions": p.exceptions,
        "duty_holder": p.duty_holder,
    }


def make_tools(bound: str) -> tuple[Callable[..., dict], Callable[..., dict]]:
    def lookup_provisions(country: str, leaves: list[str]) -> dict:
        """Provisions of a country's loaded laws that target the given content-taxonomy leaves.

        Args:
          country: two-letter country code, e.g. "MY".
          leaves: taxonomy ids of the moment, e.g. ["health.cure_claim"].
        """
        if country.strip().upper() != bound:
            return {"error": f"you assess {bound} only", "provisions": []}
        hits = retrieval.lookup(bound, leaves)
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
        # a network query (RAG Engine): keep it off the event loop the other countries share
        hits, passages = await asyncio.to_thread(retrieval.search_with_passages, bound, query, 5)
        return {"provisions": [provision_view(p) for p in hits],
                "passages": [{"law": x.law, "page": x.page, "snippet": x.snippet} for x in passages]}

    return lookup_provisions, search_law
