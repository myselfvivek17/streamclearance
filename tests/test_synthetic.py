"""AI-generated content check: stage A schema, the deterministic risk table, the mapping."""
import json
import subprocess

import pytest

from app import config, pipeline, provisions, synthetic, taxonomy
from app.pipeline import StageADraft
from app.schemas import Category, CountryReport, Inventory, Report, Segment, SyntheticCheck
from tools.build_rules import quote_on_page

from conftest import FFMPEG

STRONG = dict(likelihood="medium", signals=["0:04 the hand has six fingers"])


def _inv(**syn) -> Inventory:
    return Inventory(duration_s=20, segments=[Segment(start_s=0, end_s=20, transcript="hi")],
                     synthetic=SyntheticCheck(**syn) if syn else None)


def _rep(cc: str, risk: str = "green") -> CountryReport:
    return CountryReport(country=cc, law="x", overall_risk=risk)


# ---------------------------------------------------------------- schema

def test_stage_a_with_and_without_the_block():
    old = StageADraft.model_validate({"languages": ["en"], "segments": [{"start_s": 0, "end_s": 3}]})
    assert old.synthetic is None and old.segments[0].synthetic_signals == []
    new = StageADraft.model_validate({"segments": [{"start_s": 0, "end_s": 3, "synthetic_signals": ["warped text"]}],
                                      "synthetic": {"likelihood": "high", "signals": ["warped text"],
                                                    "label_present": False, "depicts_real_person": True,
                                                    "person_note": "a named minister"}})
    assert new.synthetic.likelihood == "high" and new.synthetic.depicts_real_person
    with pytest.raises(ValueError):
        SyntheticCheck(likelihood="certain")


def test_older_reports_still_load():
    old = {"video_id": "v", "duration_s": 10, "countries": [
        {"country": "IN", "law": "x", "overall_risk": "amber", "findings": [], "uncovered": []}],
        "inventory": {"duration_s": 10, "segments": [{"start_s": 0, "end_s": 10}]}}
    r = Report.model_validate(old)
    assert r.inventory.synthetic is None and r.countries[0].ai_content is None
    assert json.loads(r.model_dump_json())["countries"][0]["badge"] == "amber"


def test_metadata_hint_raises_likelihood():
    check = pipeline.with_metadata(SyntheticCheck(), ['file metadata: comment tag mentions "Veo"'])
    assert check.likelihood == "medium" and check.signals == check.metadata_signals
    assert pipeline.with_metadata(SyntheticCheck(likelihood="high"), ["x"]).likelihood == "high"
    assert pipeline.with_metadata(SyntheticCheck(), []) == SyntheticCheck()


@pytest.mark.skipif(not (FFMPEG and config.ffprobe_path()), reason="needs ffmpeg")
def test_ffprobe_tags_naming_an_ai_tool(tmp_path):
    from app import media

    def clip(name, *meta):
        out = tmp_path / name
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                        *meta, str(out)], check=True, timeout=60)
        return out

    assert media.ai_metadata_hints(clip("plain.mp4")) == []  # "Lavf..." is not an AI tool
    hints = media.ai_metadata_hints(clip("veo.mp4", "-metadata", "comment=Made with Google Veo 3"))
    assert hints and all("comment" in h for h in hints)


# ---------------------------------------------------------------- risk table

@pytest.fixture
def provs(real_provisions):
    return None


@pytest.mark.parametrize("cc, syn, outcome, risk, leaf", [
    ("IN", dict(likelihood="none"), "none", None, None),
    ("IN", dict(likelihood="low", signals=["slightly odd lighting"]), "none", None, None),
    ("IN", STRONG, "flag", "amber", synthetic.UNLABELLED),
    ("SG", STRONG, "check", None, synthetic.UNLABELLED),
    ("AU", STRONG, "check", None, synthetic.UNLABELLED),
    ("IN", {**STRONG, "label_present": True, "label_text": "Made with AI"}, "note", None, None),
    ("SG", {**STRONG, "label_present": True, "label_text": "Made with AI"}, "note", None, None),
    ("IN", {**STRONG, "depicts_real_person": True}, "flag", "red", synthetic.REAL_PERSON),
    ("SG", {**STRONG, "depicts_real_person": True}, "flag", "red", synthetic.REAL_PERSON),
    ("AU", {**STRONG, "depicts_real_person": True}, "check", "amber", synthetic.REAL_PERSON),
    ("MY", {**STRONG, "depicts_real_person": True}, "check", "amber", synthetic.REAL_PERSON),
    ("IN", {**STRONG, "depicts_real_person": True, "label_present": True}, "check", "amber", synthetic.REAL_PERSON),
    ("IN", {**STRONG, "likelihood": "high", "depicts_real_person": True}, "flag", "red", synthetic.REAL_PERSON),
    ("IN", {"likelihood": "low", "depicts_real_person": True}, "none", None, None),
])
def test_risk_table(provs, cc, syn, outcome, risk, leaf):
    res = synthetic.consequence(cc, SyntheticCheck(**syn), checked=True)
    assert (res.outcome, res.risk, res.leaf) == (outcome, risk, leaf)
    assert "this is AI-generated" not in res.text.lower()
    if outcome != "none":
        assert res.text.startswith("Signs of AI generation")
    assert bool(res.provision_ids) == (outcome == "flag")


def test_not_checked_country_gets_a_note_only(provs):
    rep = _rep("PK", "not_checked")
    synthetic.apply(rep, _inv(**STRONG, depicts_real_person=True))
    assert rep.ai_content.outcome == "not_checked" and not rep.findings and not rep.uncovered
    assert rep.overall_risk == "not_checked"


def test_apply_flag_adds_a_cited_video_level_finding(provs):
    rep = _rep("IN")
    synthetic.apply(rep, _inv(**STRONG))
    f = rep.findings[0]
    assert f.video_level and f.risk == "amber" and (f.start_s, f.end_s) == (0, 20)
    assert f.rule_card_ids == synthetic.PROVISIONS["IN"][synthetic.UNLABELLED]
    assert {p.id for p in rep.cited_provisions} == set(f.rule_card_ids)
    assert rep.overall_risk == "amber" and rep.badge == "amber"
    assert "label" in f.suggested_fix
    # its own category, not misinformation (the IN live report said misinformation_public_order)
    assert f.category == Category.synthetic_unlabelled
    synthetic.apply(rep, _inv(**STRONG))  # applied once only
    assert len(rep.findings) == 1


def test_apply_never_lowers_and_checks_are_listed(provs):
    rep = _rep("AU", "red")
    synthetic.apply(rep, _inv(**STRONG, depicts_real_person=True))
    assert rep.overall_risk == "red" and not rep.findings
    u = rep.uncovered[0]
    assert u.video_level and u.leaf == synthetic.REAL_PERSON and "Check manually" in u.note
    rep = _rep("MY")
    synthetic.apply(rep, _inv(**STRONG))
    assert rep.overall_risk == "green" and rep.badge == "check"  # never green-washed: grey


def test_no_block_no_change(provs):
    rep = _rep("IN")
    synthetic.apply(rep, _inv())
    assert rep.ai_content is None and not rep.findings


# ---------------------------------------------------------------- mapping

def test_taxonomy_leaves_are_policy_only():
    for leaf in (synthetic.UNLABELLED, synthetic.REAL_PERSON):
        assert taxonomy.is_leaf(leaf) and taxonomy.leaves()[leaf].policy_only
        assert leaf not in taxonomy.prompt_list()  # stage A never tags them
    assert "misinfo.synthetic_media" in taxonomy.prompt_list()


def test_mapped_provisions_exist_verified_and_quoted(real_provisions):
    for cc, by_leaf in synthetic.PROVISIONS.items():
        known = provisions.by_id(cc)
        for leaf, ids in by_leaf.items():
            assert ids, (cc, leaf)
            for pid in ids:
                p = known[pid]  # exists and its law's source is verified (else not loaded)
                assert provisions.is_verified(cc, p.law_id)
                assert p.law_status == "in_force"
                assert any(t in ("misinfo.synthetic_media", "impersonation.identity") for t in p.targets)
                law = next(x for x in provisions.laws_for(cc) if x.id == p.law_id)
                pages = {x["page"]: x["text"] for x in json.loads(law.text_path.read_text(encoding="utf-8"))}
                assert quote_on_page(p.verbatim_text, pages[p.page]), pid


@pytest.mark.parametrize("signals,flag", [
    (["The voiceover exhibits the flat inflection and robotic prosody of a text-to-speech voice."], False),
    (["The voice sounds synthetic or text-to-speech generated."], False),
    (["0:04 the hand has six fingers", "the voice sounds cloned"], True),
    (["the face morphs between frames"], True),
])
def test_voice_only_signs_are_a_manual_check_not_a_flag(provs, signals, flag):
    res = synthetic.consequence("IN", SyntheticCheck(likelihood="medium", signals=signals), checked=True)
    assert (res.outcome == "flag") == flag
    if not flag:
        assert res.outcome == "check" and res.risk is None and not res.provision_ids


def test_voice_only_flag_from_model_and_metadata_override(provs):
    s = SyntheticCheck(likelihood="medium", signals=["odd audio"], voice_only=True)
    assert synthetic.consequence("IN", s, checked=True).outcome == "check"
    m = s.model_copy(update={"metadata_signals": ["encoder: Veo"]})
    assert synthetic.consequence("IN", m, checked=True).outcome == "flag"


def test_voice_only_real_person_still_escalates(provs):
    s = SyntheticCheck(likelihood="medium", signals=["the voice sounds cloned"], depicts_real_person=True)
    assert synthetic.consequence("IN", s, checked=True).risk == "red"


def test_segment_english_field_is_optional():
    from app.schemas import Segment
    assert Segment(start_s=0, end_s=1).english == ""
    assert Segment(start_s=0, end_s=1, transcript="آج", english="Today").english == "Today"
