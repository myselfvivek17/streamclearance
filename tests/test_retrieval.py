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
