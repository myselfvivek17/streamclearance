import asyncio

from app import brain, pipeline, provisions
from app.brain import agents, verifier
from app.brain.agents import CriticVerdict, MomentAssessment
from app.schemas import CountryReport


def _inv():
    return pipeline.mock_inventory(60)


def _seg(inv, leaf):
    return next(s for s in inv.segments if leaf in s.leaves)


def test_mock_brain_my_multi_law(real_provisions):
    inv = _inv()
    rep = asyncio.run(brain.assess_country(inv, "MY", mock=True))
    CountryReport.model_validate(rep.model_dump())
    assert rep.brain == "adk" and rep.dropped_findings >= 1  # the planted invented id
    cure = [f for f in rep.findings if f.leaf == "health.cure_claim"]
    assert cure and all(pid.split("/")[1] in ("MAS1956", "MOHADV") for f in cure for pid in f.rule_card_ids)
    # MY casino promo -> the 1953 gambling Acts, never a drug or medicine provision
    casino = [f for f in rep.findings if f.leaf == "gambling.promotion"]
    assert casino and all(pid.split("/")[1] in ("CGHA1953", "BETTING1953")
                          for f in casino for pid in f.rule_card_ids)
    # no MY religion law is loaded: that moment is "not covered", never cited
    assert not [f for f in rep.findings if f.leaf == "religion.insult"]
    assert "religion.insult" in {u.leaf for u in rep.uncovered}
    assert not any("drugs" in t for p in rep.cited_provisions for t in p.targets)
    assert {law.id for law in rep.laws} >= {"OSA2025", "MAS1956", "CGHA1953", "BETTING1953"}
    assert {p.id for p in rep.cited_provisions} == {pid for f in rep.findings for pid in f.rule_card_ids}
    assert rep.badge == "red"


def test_verifier_drops_foreign_wrong_leaf_and_out_of_range(real_provisions):
    inv = _inv()
    cure, casino = _seg(inv, "health.cure_claim"), _seg(inv, "gambling.promotion")
    mas = next(p.id for p in provisions.provisions_for("MY") if p.law_id == "MAS1956")
    drugs = next(p.id for p in provisions.provisions_for("MY") if "drugs.promotion" in p.targets)
    a = lambda seg, leaf, ids, **kw: MomentAssessment(start_s=seg.start_s, end_s=seg.end_s, leaf=leaf,
                                                     covered_by=ids, **kw)
    v = verifier.verify([
        a(cure, "health.cure_claim", [mas, "IN/X/1"]),          # kept, foreign id removed
        a(casino, "gambling.promotion", [drugs]),               # drug provision for casino: dropped
        a(casino, "drugs.promotion", [drugs]),                  # leaf not tagged on moment: dropped
        a(cure, "health.cure_claim", [mas]).model_copy(update={"end_s": 999}),  # out of range
        a(casino, "gambling.promotion", []),                    # honest "not covered"
    ], "MY", inv)
    assert [x.covered_by for x in v.covered] == [[mas]]
    assert v.removed_citations == 1 and v.dropped == 3
    assert [x.leaf for x in v.not_covered] == ["gambling.promotion"]


def test_critic_verdicts():
    base = dict(start_s=0, end_s=1, leaf="health.cure_claim", covered_by=["x"], risk="red")
    found = [MomentAssessment(**base), MomentAssessment(**base), MomentAssessment(**base)]
    kept, removed, changes = agents.apply_verdicts(found, [
        CriticVerdict(index=1, action="downgrade", risk="amber", reason="education context"),
        CriticVerdict(index=2, action="remove", reason="does not cover"),
        CriticVerdict(index=7, action="remove"),  # out of range: ignored
    ])
    assert [k.risk for k in kept] == ["red", "amber"] and changes == 2
    assert kept[1].critic_note == "Critic: education context" and len(removed) == 1
    # a "downgrade" can never raise risk
    kept, _, changes = agents.apply_verdicts([MomentAssessment(**{**base, "risk": "amber"})],
                                             [CriticVerdict(index=0, action="downgrade", risk="red")])
    assert kept[0].risk == "amber" and changes == 0


def test_workflow_runs_parallel_countries(real_provisions, tiny_video):
    path, _ = tiny_video
    done = []

    async def on_done(cc, rep, err):
        done.append((cc, err))

    inv, results = asyncio.run(brain.run(path, 60.0, ["MY", "IN"], mock=True, on_done=on_done))
    assert inv.segments and sorted(done) == [("IN", None), ("MY", None)]
    assert results["MY"].brain == "adk" and results["MY"].findings
    assert results["IN"].brain == "adk" and results["IN"].findings


def test_badge_check_when_only_uncovered():
    rep = CountryReport(country="MY", law="x", overall_risk="green",
                        uncovered=[{"start_s": 0, "end_s": 1, "category": "other"}])
    assert rep.badge == "check"


def test_mock_brain_in_casino_proga_and_diabetes_not_cited_from_unverified_dmr(real_provisions):
    rep = asyncio.run(brain.assess_country(_inv(), "IN", mock=True))
    by_leaf = {f.leaf: f for f in rep.findings}
    assert {pid.split("/")[1] for pid in by_leaf["gambling.promotion"].rule_card_ids} == {"PROGA2025"}
    # DMR 1954's file is not verified against India Code: the cure claim stays uncovered
    assert "health.cure_claim" not in by_leaf
    assert any(u.leaf == "health.cure_claim" for u in rep.uncovered)
    assert not any(p.law_id == "DMR1954" for p in rep.cited_provisions)
    assert not any("drugs" in t for p in rep.cited_provisions for t in p.targets)
    laws = {law.id: law for law in rep.laws}
    assert set(laws) >= {"ITRULES2021", "PROGA2025", "DMR1954"}  # listed as a source, flagged
    assert laws["DMR1954"].provisions == 0 and not laws["DMR1954"].source_verified
    assert laws["PROGA2025"].source_verified and laws["PROGA2025"].publisher


def test_platform_obligations_attached_not_cited(real_provisions):
    from app.schemas import Inventory, Segment

    inv = Inventory(duration_s=20, segments=[
        Segment(start_s=0, end_s=10, transcript="Buy ganja from my link", leaves=["drugs.promotion"]),
        Segment(start_s=10, end_s=20, transcript="Use my casino code", leaves=["gambling.promotion"])])
    rep = asyncio.run(brain.assess_country(inv, "MY", mock=True))
    drugs = next(f for f in rep.findings if f.leaf == "drugs.promotion")
    duties = {p.id: p for p in rep.platform_obligations}
    assert duties and all(p.kind == "obligation" and p.duty_holder == "platform" for p in duties.values())
    assert drugs.obligation_ids and set(drugs.obligation_ids) <= set(duties)
    assert all(p.law_id == "OSA2025" for p in duties.values())
    # obligations are never "covered by" citations, and a casino promo gets no OSA duty
    assert not set(duties) & {pid for f in rep.findings for pid in f.rule_card_ids}
    casino = next(f for f in rep.findings if f.leaf == "gambling.promotion")
    assert casino.obligation_ids == []


def test_tools_stop_after_the_call_limit(real_provisions):
    import asyncio

    from app.brain.tools import LIMIT_REACHED, make_tools

    lookup, search = make_tools("MY", max_calls=2)
    assert lookup("MY", ["gambling.promotion"])["provisions"]
    assert "error" not in asyncio.run(search("MY", "casino promo code"))
    assert lookup("MY", ["gambling.promotion"]) == {"error": LIMIT_REACHED, "provisions": []}
    assert asyncio.run(search("MY", "casino"))["error"] == LIMIT_REACHED
    assert lookup("IN", ["gambling.promotion"])["error"] == "you assess MY only"


def test_assessor_prompt_carries_the_lookup_provisions(real_provisions):
    from app import retrieval
    from app.brain import agents

    inv = pipeline.mock_inventory(60)
    items, provs = agents.prefetch("MY", agents.moments(inv))
    gamb = next(m for m in items if "gambling.promotion" in m["leaves"])
    want = [p.id for p in retrieval.lookup("MY", gamb["leaves"])]
    assert want and gamb["candidate_provision_ids"] == want
    assert set(want) <= set(provs) and provs[want[0]]["verbatim_text"]
    prompt = agents._assess_prompt("MY", items, provs)
    assert "PROVISIONS" in prompt and want[0] in prompt


def test_country_progress_reports_assess_and_critic_steps(real_provisions):
    seen = []

    async def on_stage(key, status, detail):
        seen.append((key, status, detail))

    inv, results = asyncio.run(brain.run(None, 60, ["MY"], mock=True, on_stage=on_stage))
    steps = [d.split(":")[0] for k, st, d in seen if k == "country:MY" and st == "running"]
    assert steps == ["assess", "critic"]
