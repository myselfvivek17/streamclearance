import asyncio
import json

import pytest

from app import config, gemini, pipeline
from app.schemas import CountryReport, RuleCard

CODES = sorted(config.COUNTRIES)
UNVERIFIED = {"PK"}  # rule-card source law not verified against its official source
VERIFIED_CODES = [c for c in CODES if c not in UNVERIFIED]


def _card(id, category="religious_offence", applies_to="platform", country="PK"):
    return RuleCard(id=id, country=country, law="PECA", section="s. 26A", page=3, category=category,
                    conduct="c", applies_to=applies_to, platform_duty="d", penalty="p", quote="q")


@pytest.mark.parametrize("code", VERIFIED_CODES)
def test_mock_stage_b_valid_and_cites_only_own_cards(code):
    cards, status = pipeline.load_cards(code)
    assert cards and status == "draft/unreviewed"
    assert config.COUNTRIES[code].dryrun_rules_path.exists()  # generated on demand
    inv = pipeline.mock_inventory(60)
    rep = pipeline.stage_b(inv, code, cards, rules_status=status, mock=True)
    CountryReport.model_validate(rep.model_dump())
    own = {c.id for c in cards}
    assert rep.findings, "mock inventory should produce findings"
    assert all(cid in own for f in rep.findings for cid in f.rule_card_ids)
    assert all(cid.startswith(code + "-") for f in rep.findings for cid in f.rule_card_ids)
    assert rep.dropped_findings >= 1  # the planted hallucinated id
    assert {c.id for c in rep.cited_cards} == {cid for f in rep.findings for cid in f.rule_card_ids}
    assert rep.overall_risk == pipeline.worst_risk(rep.findings)
    assert rep.rules_status == "draft/unreviewed" and "Advisory only" in rep.disclaimer


def test_reviewed_rules_preferred_over_dryrun():
    path = config.COUNTRIES["AU"].rules_path
    card = _card("AU-001", country="AU").model_dump(mode="json")
    path.write_text(json.dumps([card | {"needs_review": False}]))
    cards, status = pipeline.load_cards("AU")
    assert [c.id for c in cards] == ["AU-001"] and status == "reviewed"
    path.write_text(json.dumps([card]))
    assert pipeline.load_cards("AU")[1] == "draft/unreviewed"


@pytest.mark.parametrize("mock", [True, False])
def test_unverified_source_gives_no_cards_and_no_model_call(monkeypatch, mock):
    path = config.COUNTRIES["PK"].rules_path  # even a reviewed rule-card file is ignored
    path.write_text(json.dumps([_card("PK-001").model_dump(mode="json") | {"needs_review": False}]))
    cards, status = pipeline.load_cards("PK")
    assert cards == [] and status == "draft/unreviewed"
    assert not config.COUNTRIES["PK"].dryrun_rules_path.exists()
    monkeypatch.setattr(gemini, "generate_json", lambda *a, **k: pytest.fail("model called"))
    rep = pipeline.stage_b(pipeline.mock_inventory(60), "PK", cards, rules_status=status, mock=mock)
    assert rep.findings == [] and rep.cited_cards == [] and rep.overall_risk == "green"
    assert rep.notes == [pipeline.UNVERIFIED_NOTE] and rep.uncovered


def test_real_stage_b_drops_invented_and_foreign_ids(monkeypatch):
    cards = [_card("PK-001"), _card("PK-002", "violence_or_gore", applies_to="creator")]
    base = dict(category="religious_offence", suggested_fix="f", rationale="r")
    seen = {}

    def fake(prompt, schema, **kw):
        seen["prompt"] = prompt
        return [
            dict(base, start_s=1, end_s=3, rule_card_ids=["PK-001"], risk="amber"),
            dict(base, start_s=4, end_s=6, rule_card_ids=["PK-002"], risk="red",
                 category="violence_or_gore"),
            dict(base, start_s=4, end_s=6, rule_card_ids=["PK-002"], risk="red"),  # category mismatch
            dict(base, start_s=1, end_s=3, rule_card_ids=["PK-777"], risk="red"),   # invented
            dict(base, start_s=1, end_s=3, rule_card_ids=["IN-001"], risk="red"),   # other country
            dict(base, start_s=50, end_s=70, rule_card_ids=["PK-001"], risk="red"),  # past end
            dict(base, start_s=1, end_s=3, rule_card_ids=[], risk="red"),           # no citation
            dict(base, start_s=1, end_s=3, rule_card_ids=["PK-001"], risk="green"),
            {"garbage": True},
        ]

    monkeypatch.setattr(gemini, "generate_json", fake)
    rep = pipeline.stage_b(pipeline.mock_inventory(20), "PK", cards, mock=False)
    assert "PK-001" in seen["prompt"] and "fasting" in seen["prompt"]
    assert [f.rule_card_ids for f in rep.findings] == [["PK-001"], ["PK-002"]]
    assert rep.overall_risk == "red" and rep.dropped_findings == 7
    assert [f.creator_liability for f in rep.findings] == [False, True]
    assert "applies to the person who posts" in rep.risk_framing


def test_no_findings_is_green():
    rep = pipeline.stage_b(pipeline.Inventory(duration_s=5), "AU", [_card("AU-001")], mock=True)
    assert rep.overall_risk == "green" and rep.findings == []


def test_run_countries_concurrent_cap_and_errors(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_CONCURRENCY", 2)
    active, peak = 0, 0
    real_stage_b = pipeline.stage_b

    def slow_stage_b(inv, cc, cards, **kw):
        nonlocal active, peak
        import time
        active += 1
        peak = max(peak, active)
        time.sleep(0.05)
        active -= 1
        if cc == "SG":
            raise RuntimeError("boom")
        return real_stage_b(inv, cc, cards, **kw)

    monkeypatch.setattr(pipeline, "stage_b", slow_stage_b)
    events = []

    async def on_done(cc, rep, err):
        events.append((cc, rep is not None, err))

    out = asyncio.run(pipeline.run_countries(pipeline.mock_inventory(30), CODES, mock=True,
                                             on_done=on_done))
    assert peak <= 2
    assert isinstance(out["SG"], RuntimeError)
    assert all(isinstance(out[c], CountryReport) for c in CODES if c != "SG")
    assert ("SG", False, "RuntimeError: boom") in events
