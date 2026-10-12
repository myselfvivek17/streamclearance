"""Findings grouped by moment: one row per moment, worst risk first, flat findings kept."""
import json

from app.schemas import CountryReport, Finding, Report


def _f(start, end, leaf, risk, ids, **kw):
    return Finding(start_s=start, end_s=end, category="other", rule_card_ids=ids, risk=risk,
                   suggested_fix=f"fix {leaf}", rationale="r", leaf=leaf, **kw)


def test_au_cure_and_stop_medicine_are_one_moment():
    rep = CountryReport(country="AU", law="x", overall_risk="red", findings=[
        _f(3.5, 11.5, "health.cure_claim", "red", ["AU/TGACODE2021/s9(1)", "AU/TGACODE2021/s28"]),
        _f(3.5, 11.5, "health.misinformation", "amber", ["AU/TGACODE2021/s9(3)(b)-(e)"]),
        _f(11.5, 18.0, "gambling.promotion", "red", ["AU/IGA2001/s61BA(1)"]),
    ])
    ms = rep.moments
    assert [(m.start_s, m.end_s, m.risk, len(m.issues)) for m in ms] == [(3.5, 11.5, "red", 2), (11.5, 18.0, "red", 1)]
    # each issue keeps its own sections and fix, worst first
    assert [i.leaf for i in ms[0].issues] == ["health.cure_claim", "health.misinformation"]
    assert ms[0].issues[1].rule_card_ids == ["AU/TGACODE2021/s9(3)(b)-(e)"]
    dumped = json.loads(rep.model_dump_json())
    assert len(dumped["findings"]) == 3 and len(dumped["moments"]) == 2  # flat list kept


def test_overlapping_ranges_merge_and_whole_video_stands_apart():
    rep = CountryReport(country="IN", law="x", overall_risk="red", findings=[
        _f(0, 5, "health.cure_claim", "amber", ["a"]),
        _f(4, 9, "health.misinformation", "red", ["b"]),
        _f(9, 12, "gambling.promotion", "amber", ["c"]),  # touches, does not overlap
        _f(0, 20, "misinfo.synthetic_unlabelled", "amber", ["d"], video_level=True),
    ])
    ms = rep.moments
    assert [(m.start_s, m.end_s, m.risk, m.video_level) for m in ms] == [
        (0, 9, "red", False), (9, 12, "amber", False), (0, 20, "amber", True)]


def test_old_reports_gain_moments_on_load():
    old = {"video_id": "v", "duration_s": 20, "countries": [{"country": "AU", "law": "x", "overall_risk": "red",
           "findings": [_f(3.5, 11.5, "health.cure_claim", "red", ["a"]).model_dump(mode="json"),
                        _f(3.5, 11.5, "health.misinformation", "red", ["b"]).model_dump(mode="json")]}]}
    assert len(Report.model_validate(old).countries[0].moments) == 1
