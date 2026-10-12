"""Rules status reflects the real files: "reviewed" only after a person signed off everything,
never set automatically, never because of the dry-run path."""
import json

from fastapi.testclient import TestClient

from app import config, main, pipeline, provisions
from app.main import app


def _country(client, cc):
    return next(c for c in client.get("/api/countries").json() if c["code"] == cc)


def test_my_status_comes_from_its_provisions(real_provisions):
    # MY has provision files (all drafts) and data/rules/my.json: the provisions win, so MY is
    # a draft because its provisions are drafts, not because of the dry-run cards
    assert provisions.has_provisions("MY")
    cards, status = pipeline.load_cards("MY")
    assert status == "draft/unreviewed" and len(cards) == len(provisions.provisions_for("MY"))
    assert not config.COUNTRIES["MY"].dryrun_rules_path.exists()
    for p in provisions.provisions_for("MY"):
        p.review.status = "reviewed"  # a person signed off every one (simulated)
    try:
        assert pipeline.load_cards("MY")[1] == "reviewed"
        with TestClient(app) as client:
            c = _country(client, "MY")
            assert c["rules_status"] == "reviewed" and c["draft"] is False
    finally:
        provisions.reload()


def test_one_unreviewed_provision_keeps_the_country_draft(real_provisions):
    provs = provisions.provisions_for("MY")
    for p in provs[1:]:
        p.review.status = "reviewed"
    try:
        assert pipeline.load_cards("MY")[1] == "draft/unreviewed"
    finally:
        provisions.reload()


def _write_rules(cc, needs_review):
    cards = json.loads((config.DATA_DIR / "rules" / "my.json").read_text(encoding="utf-8"))
    for c in cards:
        c["needs_review"] = needs_review
    config.COUNTRIES[cc].rules_path.write_text(json.dumps(cards), encoding="utf-8")


def test_real_rules_file_is_used_not_the_dry_run():
    # no provisions loaded here (conftest): MY falls back to its rule-card file
    _write_rules("MY", needs_review=True)
    assert pipeline.load_cards("MY")[1] == "draft/unreviewed"
    _write_rules("MY", needs_review=False)
    cards, status = pipeline.load_cards("MY")
    assert status == "reviewed" and cards
    assert not config.COUNTRIES["MY"].dryrun_rules_path.exists()
    assert main._rules_status(config.COUNTRIES["MY"]) == "reviewed"


def test_never_reviewed_automatically(real_provisions):
    """The API says "reviewed" exactly when every provision file entry says so."""
    with TestClient(app) as client:
        for cc in ("IN", "AU", "MY", "SG"):
            signed = all(p.review.status == "reviewed" for p in provisions.provisions_for(cc))
            assert _country(client, cc)["rules_status"] == ("reviewed" if signed else "draft/unreviewed")
