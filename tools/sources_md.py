"""Write docs/SOURCES.md: where every law document comes from, from data/laws/registry.json.

    python -m tools.sources_md

tests/test_laws.py checks the committed file matches the registry.
"""
from __future__ import annotations

from app import config, provisions

PATH = config.ROOT / "docs" / "SOURCES.md"

HEAD = """# Law sources

Every document in `data/laws/raw/` and where it was taken from. Generated from
`data/laws/registry.json` by `python -m tools.sources_md`; do not edit by hand.

A law is **verified** when the file held matches the official publisher's file
(same sha256, downloaded again on the date shown). Only verified laws are
indexed into the search, retrieved or cited in reports; unverified ones are
listed as sources and flagged.

| Country | Law | Version | Publisher | Official file | sha256 | Authoritative language | Verified |
|---|---|---|---|---|---|---|---|
"""


def _cell(text: str | None) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def render() -> str:
    rows, notes = [], []
    for law in provisions.registry():
        link = f"[{law.pdf}]({law.source_url})" if law.source_url else law.pdf
        if law.source_page and law.source_page != law.source_url:
            link += f" ([page]({law.source_page}))"
        verified = f"yes ({law.retrieved_at})" if law.source_verified else "**no**"
        rows.append("| " + " | ".join(_cell(x) for x in (
            law.country, law.title, law.version, law.publisher, link, f"`{(law.sha256 or '')[:12]}`",
            law.authoritative_language, verified)) + " |")
        for label, text in (("Not verified", law.unverified_reason), ("Text source", law.text_source),
                            ("Superseded by", law.superseded_by)):
            if text:
                notes.append(f"- **{law.key}**, {label.lower()}: {text}")
    out = HEAD + "\n".join(rows) + "\n"
    if notes:
        out += "\n## Notes\n\n" + "\n".join(notes) + "\n"
    return out


if __name__ == "__main__":
    PATH.write_text(render(), encoding="utf-8")
    print(f"wrote {PATH.relative_to(config.ROOT)}")
