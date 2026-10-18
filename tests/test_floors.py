"""Severity floors (app/floors.py): raise only, only on a genuinely matching cited provision."""
import pytest

from app import floors
from app.schemas import CountryReport, Finding, Inventory, Segment

GAMBLING = "Use code WIN50 at LuckySpin casino for fifty free spins."
CURE = "This herbal tea cured my diabetes in two weeks."
STOP = "This herbal tea cured my diabetes. You can stop taking your medicine."
TIRED = "This herbal tea cures tiredness."


def _case(cc, leaf, risk, ids, text, **kw):
    inv = Inventory(duration_s=20, segments=[Segment(start_s=0, end_s=10, transcript=text, leaves=[leaf])])
    f = Finding(start_s=0, end_s=10, category="other", rule_card_ids=ids, risk=risk, suggested_fix="x",
                rationale="y", leaf=leaf, **kw)
    rep = CountryReport(country=cc, law="x", overall_risk=risk, findings=[f])
    n = floors.apply(rep, inv)
    return rep, rep.findings[0], n


@pytest.mark.parametrize("cc, leaf, risk, ids, text, expected", [
    # MY casino promo, amber in one run: CGHA s4(1)(g) binds the creator, in force -> red
    ("MY", "gambling.promotion", "amber", ["MY/CGHA1953/s4(1)(g)"], GAMBLING, "red"),
    ("SG", "gambling.promotion", "amber", ["SG/GCA2022/s85(1)"], GAMBLING, "red"),
    ("IN", "gambling.promotion", "amber", ["IN/ITRULES2021/r3(1)(b)(x)"], GAMBLING, "red"),
    # serious disease cure claim under an in-force advertising law -> red
    ("MY", "health.cure_claim", "amber", ["MY/MAS1956/s3(1)"], CURE, "red"),
    ("AU", "health.cure_claim", "amber", ["AU/TGACODE2021/s9(1)"], CURE, "red"),
    # a guideline is not an in-force law: no floor
    ("MY", "health.cure_claim", "amber", ["MY/MOHADV/4.4(a)"], CURE, "amber"),
    # not a serious disease and no stop-medicine line: no floor
    ("MY", "health.cure_claim", "amber", ["MY/MAS1956/s3(1)"], TIRED, "amber"),
    # stop your medicine -> red for health misinformation too
    ("IN", "health.misinformation", "amber", ["IN/ITRULES2021/r3(1)(b)(v)"], STOP, "red"),
    ("IN", "health.misinformation", "amber", ["IN/ITRULES2021/r3(1)(b)(v)"], CURE, "amber"),
    # MY stop-your-medicine flipped red/amber between runs: either MOH guideline the
    # assessor cites for it gives red (5.11: "discontinue ... prescribed medicines")
    ("MY", "health.misinformation", "amber", ["MY/MOHADV/5.11"], STOP, "red"),
    ("MY", "health.misinformation", "amber", ["MY/MOHADV/4.11"], STOP, "red"),
    ("AU", "health.misinformation", "amber", ["AU/TGACODE2021/s9(3)(b)-(e)"], STOP, "red"),
    # no stop-medicine line: the guideline gives no floor
    ("MY", "health.misinformation", "amber", ["MY/MOHADV/5.11"], CURE, "amber"),
    # never lowers
    ("MY", "gambling.promotion", "red", ["MY/CGHA1953/s4(1)(g)"], GAMBLING, "red"),
    # a leaf without a floor
    ("IN", "religion.insult", "amber", ["IN/ITRULES2021/r3(1)(b)(x)"], GAMBLING, "amber"),
])
def test_floor_table(real_provisions, cc, leaf, risk, ids, text, expected):
    rep, f, n = _case(cc, leaf, risk, ids, text)
    assert f.risk == expected and n == (expected != risk)
    assert rep.overall_risk == expected
    assert bool(f.floor_note) == (expected != risk)
    if f.floor_note:
        assert "fixed rule" in f.floor_note and ids[0] in f.floor_note


def test_platform_only_or_mismatched_citation_does_not_raise(real_provisions):
    # IN r3(3)(a)(i)(IV) binds platforms and targets synthetic media, not gambling
    _, f, n = _case("IN", "gambling.promotion", "amber", ["IN/ITRULES2021/r3(3)(a)(i)(IV)"], GAMBLING)
    assert f.risk == "amber" and n == 0


def test_exception_blocks_the_floor(real_provisions):
    _, f, _ = _case("MY", "gambling.promotion", "amber", ["MY/CGHA1953/s4(1)(g)"], GAMBLING,
                    exception_applies=True, exception_reason="news report")
    assert f.risk == "amber"


def test_legacy_findings_without_leaf_untouched(real_provisions):
    inv = Inventory(duration_s=10, segments=[Segment(start_s=0, end_s=10, transcript=GAMBLING)])
    f = Finding(start_s=0, end_s=10, category="illegal_goods_or_gambling", rule_card_ids=["MY-1"],
                risk="amber", suggested_fix="x", rationale="y")
    rep = CountryReport(country="MY", law="x", overall_risk="amber", findings=[f])
    assert floors.apply(rep, inv) == 0 and f.risk == "amber"


def test_stop_medicine_phrases():
    for t in ["stop taking your insulin", "you can quit your medication", "throw away your pills",
              "no need for metformin anymore"]:
        assert floors.STOP_MEDICINE.search(t), t
    for t in ["stop scrolling and buy", "medicine cabinet makeover"]:
        assert not floors.STOP_MEDICINE.search(t), t


def test_stop_medicine_is_red_in_every_run(real_provisions):
    """The same MY moment, as assessed in two live runs (amber on 4.11, red on 5.11),
    ends red both times; the cure claim at that moment stays red on the in-force law."""
    for ids, risk in [(["MY/MOHADV/4.11"], "amber"), (["MY/MOHADV/5.11"], "red")]:
        rep, f, _ = _case("MY", "health.misinformation", risk, ids, STOP)
        assert f.risk == "red" and rep.overall_risk == "red"
