import pytest

from app import provisions, retrieval
from app.schemas import Provision


def _prov(pid, targets, *, law="OSA2025", status="in_force", review="draft", text="", summary=""):
    return Provision(
        id=f"MY/{law}/{pid}", country="MY", law_id=law, law_title=law, law_status=status,
        source_pdf="x.pdf", page=1, section=pid, verbatim_text=text or f"text {pid}",
        summary=summary, targets=targets, duty_holder="platform", review={"status": review})


@pytest.fixture
def store(monkeypatch):
    data = {"MY": [
        _prov("drugs", ["drugs.promotion"], text="content that promotes the use or sale of dangerous drugs"),
        _prov("casino", ["gambling.promotion"], law="CGH1953", text="advertises a common gaming house"),
        _prov("allgambling", ["gambling"], law="BET1953", text="betting and lotteries of any kind"),
        _prov("tipsbill", ["gambling.betting_tips"], law="BILL", status="bill"),
        _prov("cure", ["health.cure_claim"], law="MAS1956", review="reviewed",
              text="advertisement for the purpose of the prevention or treatment of diabetes",
              summary="Bans ads claiming to cure listed diseases such as diabetes"),
        _prov("guide", ["health.cure_claim", "health.medicine_advertising"], law="MOHADV",
              status="guideline", text="claims to cure diabetes, cancer"),
    ], "IN": [_prov("in-cure", ["health.cure_claim"]).model_copy(update={"country": "IN", "id": "IN/X/1"})]}
    monkeypatch.setattr(provisions, "_cache", data)
    retrieval.reset()
    yield data
    retrieval.reset()
    monkeypatch.setattr(provisions, "_cache", None)


def test_casino_promo_gets_gambling_only(store):
    hits = retrieval.lookup("MY", ["gambling.promotion"])
    ids = [p.id for p in hits]
    assert ids == ["MY/CGH1953/casino", "MY/BET1953/allgambling"]  # exact first, then parent-level
    assert not any("drugs" in t for p in hits for t in p.targets)


def test_parent_query_and_ranking(store):
    ids = [p.id for p in retrieval.lookup("MY", ["gambling"])]
    assert ids[0] == "MY/BET1953/allgambling"
    assert set(ids) == {"MY/BET1953/allgambling", "MY/CGH1953/casino", "MY/BILL/tipsbill"}
    assert ids[-1] == "MY/BILL/tipsbill"  # bill ranks last
    cure = [p.id for p in retrieval.lookup("MY", ["health.cure_claim"])]
    assert cure == ["MY/MAS1956/cure", "MY/MOHADV/guide"]  # reviewed + in force before guideline
    assert retrieval.lookup("MY", ["religion.insult"]) == []
    assert [p.country for p in retrieval.lookup("IN", ["health.cure_claim"])] == ["IN"]
    assert len(retrieval.lookup("MY", ["gambling", "health", "drugs"], cap=3)) == 3


def test_search_bm25(store):
    hits = retrieval.search("MY", "this herbal drink cures diabetes", k=3)
    assert hits and hits[0].law_id in ("MAS1956", "MOHADV")
    assert all(p.country == "MY" for p in hits)
    assert "MY/MAS1956/cure" in [p.id for p in hits]
    assert retrieval.search("MY", "casino gaming house")[0].id == "MY/CGH1953/casino"
    assert retrieval.search("MY", "zzzz unrelated") == []


def test_cards_derive_from_provisions(real_provisions):
    from app import pipeline

    cards, status = pipeline.load_cards("MY")
    assert cards and status == "draft/unreviewed"
    ids = {p.id for p in provisions.provisions_for("MY")}
    assert {c.id for c in cards} == ids
    assert all(len(c.quote) <= 300 and c.country == "MY" for c in cards)
    rep = pipeline.stage_b(pipeline.mock_inventory(60), "MY", cards, rules_status=status, mock=True)
    assert all(cid in ids for f in rep.findings for cid in f.rule_card_ids)


def test_real_store_casino_and_diabetes_per_country(real_provisions):
    my_casino = retrieval.lookup("MY", ["gambling.promotion"])
    assert my_casino and {p.law_id for p in my_casino} <= {"CGHA1953", "BETTING1953"}
    in_casino = retrieval.lookup("IN", ["gambling.promotion"])
    assert {p.law_id for p in in_casino} >= {"PROGA2025"}
    assert any(p.id in ("IN/PROGA2025/s6", "IN/PROGA2025/s9(2)") for p in in_casino)
    in_cure = retrieval.lookup("IN", ["health.cure_claim"])
    assert in_cure and in_cure[0].law_id == "DMR1954"
    assert "diabetes" in " ".join(p.verbatim_text.lower() for p in in_cure)
    for cc in ("MY", "IN"):  # no drug or medicine provision for gambling, and vice versa
        assert not any(t.startswith(("drugs", "health")) for p in retrieval.lookup(cc, ["gambling.promotion"])
                       for t in p.targets)
        assert all(p.kind == "content" for p in retrieval.lookup(cc, ["gambling.promotion", "health.cure_claim"]))


def test_obligations_ranked_specific_first(real_provisions):
    duties = retrieval.obligations("MY", ["fraud.financial"])
    assert duties and all(p.kind == "obligation" for p in duties)
    assert len(duties[0].targets) <= len(duties[-1].targets)
    assert retrieval.obligations("MY", ["health.cure_claim"]) == []


def test_real_store_au_sg(real_provisions):
    au_cure = retrieval.search("AU", "herbal tea cures diabetes", k=3)
    assert au_cure and all(p.law_id == "TGACODE2021" for p in au_cure)
    assert {p.law_id for p in retrieval.lookup("AU", ["health.cure_claim"])} == {"TGACODE2021"}
    au_casino = retrieval.lookup("AU", ["gambling.promotion"])
    assert au_casino and {p.law_id for p in au_casino} == {"IGA2001"}
    assert any(p.id.startswith("AU/IGA2001/s61BA") for p in au_casino)  # Part 7A advertising ban
    assert "IGA2001" in {p.law_id for p in retrieval.search("AU", "casino promo code free bets")}
    sg_casino = retrieval.search("SG", "online casino referral code", k=5)
    assert sg_casino and {p.law_id for p in sg_casino} == {"GCA2022"}
    assert {p.law_id for p in retrieval.lookup("SG", ["gambling.promotion"])} == {"GCA2022"}
    # SG has no health-advertising law: nothing is forced
    assert retrieval.lookup("SG", ["health.cure_claim"]) == []
    assert retrieval.search("SG", "herbal tea cures diabetes") == []
    duties = retrieval.obligations("SG", ["harassment.bullying"])
    assert duties and all(p.kind == "obligation" and p.law_id == "OSRA2025" for p in duties)
    duties_bm25 = retrieval.BM25([p for p in provisions.provisions_for("SG") if p.kind == "obligation"])
    top = duties_bm25.search("harmful online content what must platforms do", 3)
    assert top and all(p.law_id == "OSRA2025" for _, p in top)
    for cc in ("AU", "SG"):
        assert not any(t.startswith(("drugs", "health")) for p in retrieval.lookup(cc, ["gambling.promotion"])
                       for t in p.targets)
