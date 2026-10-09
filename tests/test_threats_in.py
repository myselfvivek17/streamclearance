"""India: threats, extortion and sextortion (Bharatiya Nyaya Sanhita, 2023)."""
import json

from app import provisions, retrieval, taxonomy
from tools.build_rules import quote_on_page

BNS = "IN/BNS2023"


def _pages():
    law = provisions.get_law(BNS)
    return {p["page"]: p["text"] for p in json.loads(law.text_path.read_text(encoding="utf-8"))}


def test_bns_is_verified_from_an_official_host():
    law = provisions.get_law(BNS)
    assert law.source_verified and law.status == "in_force"
    assert law.source_url.startswith("https://www.mha.gov.in/")
    assert "No. 45 of 2023" in law.title


def test_bns_provisions_loaded_and_quotes_verified(real_provisions):
    loaded = {p.id: p for p in provisions.provisions_for("IN") if p.law_id == "BNS2023"}
    for ref in ("s308(1)", "s308(2)", "s308(3)", "s351(1)", "s351(2)", "s351(3)", "s351(4)"):
        assert f"{BNS}/{ref}" in loaded, ref
    pages = _pages()
    text = " ".join(pages.values())
    for p in loaded.values():
        assert quote_on_page(p.verbatim_text, pages[p.page]), p.id
        assert p.duty_holder == "creator"
        # penalties are quoted from the statute (definitions cite the punishing sub-section)
        penalty = p.consequence.penalty.split(": ", 1)[-1]
        assert quote_on_page(penalty, text), p.id


def test_extortion_leaves_exist():
    for leaf in ("harassment.extortion", "harassment.sextortion"):
        assert taxonomy.is_leaf(leaf)
        assert taxonomy.category_of(leaf).value == "harassment_or_bullying"


def test_extortion_maps_to_bns_308_and_351(real_provisions):
    for leaf in ("harassment.extortion", "harassment.sextortion"):
        ids = {p.id for p in retrieval.lookup("IN", [leaf])}
        assert {f"{BNS}/s308(1)", f"{BNS}/s308(3)", f"{BNS}/s351(1)"} <= ids, leaf
    threats = {p.id for p in retrieval.lookup("IN", ["harassment.threats"])}
    assert f"{BNS}/s351(1)" in threats


def test_stage_a_guidance_tags_sextortion_as_extortion():
    from app.pipeline import STAGE_A_SYSTEM
    prompt = " ".join(STAGE_A_SYSTEM.split())
    assert '"harassment.sextortion", "harassment.extortion" and "harassment.threats"' in prompt
    assert "only if such images are shown or said to have already been shared" in prompt
