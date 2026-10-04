import pytest
from pydantic import ValidationError

from app.schemas import Category, CountryReport, Finding, Inventory, Report, RuleCard, Segment


def _card(**kw):
    base = dict(
        id="IN-001", country="IN", law="IT Rules 2021", section="Rule 3(1)(b)(ii)", page=6,
        category="sexual_or_nudity", conduct="Obscene or pornographic content",
        applies_to="platform", platform_duty="Remove within 24h of complaint",
        penalty="Loss of safe harbour", quote="is obscene, pornographic, paedophilic",
    )
    return RuleCard(**{**base, **kw})


def test_rule_card_round_trip():
    card = _card()
    assert card.needs_review is True
    assert RuleCard.model_validate_json(card.model_dump_json()) == card


def test_rule_card_rejects_bad_values():
    with pytest.raises(ValidationError):
        _card(country="US")
    with pytest.raises(ValidationError):
        _card(category="not_a_category")
    with pytest.raises(ValidationError):
        _card(quote="x" * 301)
    with pytest.raises(ValidationError):
        _card(applies_to="everyone")


def test_report_round_trip():
    inv = Inventory(duration_s=60, languages=["en"], segments=[
        Segment(start_s=0, end_s=5, transcript="hi", categories=[Category.other]),
    ])
    finding = Finding(start_s=1, end_s=4, category="other", rule_card_ids=["IN-001"],
                      risk="amber", suggested_fix="Blur", rationale="Because")
    report = Report(video_id="v1", duration_s=60, countries=[
        CountryReport(country="IN", law="IT Rules 2021", overall_risk="amber", findings=[finding]),
    ])
    assert Inventory.model_validate_json(inv.model_dump_json()) == inv
    assert Report.model_validate_json(report.model_dump_json()) == report


def test_finding_requires_ordered_timestamps_and_citation():
    with pytest.raises(ValidationError):
        Finding(start_s=5, end_s=1, category="other", rule_card_ids=["X"],
                risk="red", suggested_fix="", rationale="")
    with pytest.raises(ValidationError):
        Finding(start_s=0, end_s=1, category="other", rule_card_ids=[],
                risk="red", suggested_fix="", rationale="")
