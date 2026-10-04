from app.schemas import Finding, RuleCard
from app.validate import validate_findings

CARD = RuleCard(
    id="IN-001", country="IN", law="IT Rules 2021", section="Rule 3(1)(b)", page=6,
    category="hate_or_incitement", conduct="c", applies_to="platform",
    platform_duty="d", penalty="p", quote="q",
)


def _finding(start, end, ids=("IN-001",)):
    return Finding(start_s=start, end_s=end, category="hate_or_incitement",
                   rule_card_ids=list(ids), risk="red", suggested_fix="f", rationale="r")


def test_keeps_valid_finding():
    good = _finding(1, 5)
    assert validate_findings([good], [CARD], duration_s=10) == [good]


def test_rejects_unknown_card_id():
    assert validate_findings([_finding(1, 5, ids=("IN-999",))], [CARD], 10) == []
    assert validate_findings([_finding(1, 5, ids=("IN-001", "IN-999"))], [CARD], 10) == []


def test_rejects_timestamp_past_duration():
    assert validate_findings([_finding(8, 12)], [CARD], duration_s=10) == []
    assert validate_findings([_finding(11, 11)], [CARD], duration_s=10) == []
