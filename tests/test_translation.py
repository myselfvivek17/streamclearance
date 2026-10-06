import asyncio

from app import config, gemini, pipeline
from app.schemas import CountryReport, DriftCheck


def _reports(codes):
    inv = pipeline.mock_inventory(60)
    reps = {}
    for cc in codes:
        cards, status = pipeline.load_cards(cc)
        reps[cc] = pipeline.stage_b(inv, cc, cards, rules_status=status, mock=True)
    return inv, reps


def test_mock_drift_one_increased_and_english_skipped():
    inv, reps = _reports(["AU", "IN", "MY", "PK", "SG"])
    assert asyncio.run(pipeline.run_translation(inv, reps, mock=True)) == []
    assert reps["AU"].translation == []  # English audience: skipped
    for cc in ["IN", "MY", "PK", "SG"]:
        assert reps[cc].translation, cc
        langs = {d.language for d in reps[cc].translation}
        assert langs == set(config.COUNTRIES[cc].drift_languages)
        for d in reps[cc].translation:
            DriftCheck.model_validate(d.model_dump())
            assert d.original.strip()
    drifts = [d.drift for r in reps.values() for d in r.translation]
    assert drifts.count("increased") == 1
    CountryReport.model_validate(reps["IN"].model_dump())


def test_segments_order_findings_then_idioms_then_rest(monkeypatch):
    inv, reps = _reports(["IN"])
    segs = pipeline.flagged_segments(inv, reps["IN"])
    flagged_starts = {f.start_s for f in reps["IN"].findings}
    assert all(s.transcript for s in segs)
    n_flagged = sum(s.start_s in flagged_starts for s in segs)
    assert n_flagged and all(s.start_s in flagged_starts for s in segs[:n_flagged])  # findings first
    nxt = segs[n_flagged]
    assert nxt.idioms and nxt.start_s not in flagged_starts  # "a killer deal"
    rest = segs[n_flagged + 1:]
    assert rest and all(not s.idioms and s.start_s not in flagged_starts for s in rest)
    assert [s.start_s for s in rest] == sorted(s.start_s for s in rest)  # no claims/leaves: in order
    monkeypatch.setattr(config, "MAX_DRIFT_SEGMENTS", 1)
    assert len(pipeline.flagged_segments(inv, reps["IN"])) == 1


def test_clean_country_with_speech_still_gets_drift():
    inv, reps = _reports(["MY", "SG"])
    for rep in reps.values():
        rep.findings, rep.overall_risk, rep.uncovered = [], "green", []
    assert asyncio.run(pipeline.run_translation(inv, reps, mock=True)) == []
    assert {d.language for d in reps["SG"].translation} == {"Mandarin Chinese", "Malay", "Tamil"}
    assert reps["MY"].translation and reps["MY"].translation_note == ""
    # the mock marks the first pair as riskier: a clean country turns amber
    riskier = [cc for cc, r in reps.items() if any(d.drift == "increased" for d in r.translation)]
    assert len(riskier) == 1
    rep = reps[riskier[0]]
    assert rep.overall_risk == "green" and rep.badge == "amber"
    assert rep.model_dump(mode="json")["badge"] == "amber"


def test_no_speech_gives_explicit_note():
    inv, reps = _reports(["MY", "AU"])
    for s in inv.segments:
        s.transcript = ""
    assert not pipeline.has_speech(inv)
    assert asyncio.run(pipeline.run_translation(inv, reps, mock=True)) == []
    assert reps["MY"].translation == [] and reps["MY"].translation_note == "No speech to translate."
    assert reps["AU"].translation_note == ""  # English audience: not applicable


def test_red_country_stays_red_with_increased_drift():
    inv, reps = _reports(["MY"])
    rep = reps["MY"]
    assert rep.overall_risk == "red"
    rep.translation = pipeline._mock_drift(inv.segments[1:2], "Malay", increase_first=True)
    assert rep.badge == "red"


def test_real_drift_translates_back_translates_blind_and_judges(monkeypatch):
    inv, reps = _reports(["IN"])
    calls = []
    n = len(pipeline.flagged_segments(inv, reps["IN"]))

    def fake(prompt, schema, **kw):
        calls.append((prompt, kw))
        if len(calls) == 1:
            return [{"index": i, "language": "Hindi", "text": f"HINDI{i}"} for i in range(n)]
        if len(calls) == 2:
            return [{"index": k, "text": f"BACK{k}"} for k in range(n)]
        return [{"index": 0, "drift": "increased", "explanation": "x"}] + \
               [{"index": k, "drift": "none", "explanation": "y"} for k in range(1, n)]

    monkeypatch.setattr(gemini, "generate_json", fake)
    out = pipeline.translation_check(inv, "IN", ["Hindi"], report=reps["IN"], mock=False)
    assert len(calls) == 3
    assert "Hindi" in calls[0][0] and "fasting" in calls[0][0]
    assert "fasting" not in calls[1][0] and "HINDI0" in calls[1][0]  # blind back-translation
    assert "Hindi" not in calls[1][0].split("\n", 1)[1]
    assert "BACK0" in calls[2][0] and "religious_offence" in calls[2][0]
    assert calls[0][1]["thinking"] == calls[1][1]["thinking"] == "low"
    assert calls[2][1].get("thinking") is None  # the judge keeps the default
    first = next(d for d in out if d.translated == "HINDI0")
    assert first.drift == "increased" and first.back_translated == "BACK0"
    assert [d.start_s for d in out] == sorted(d.start_s for d in out)  # shown in video order


def test_all_languages_of_a_country_in_three_calls(monkeypatch):
    inv, reps = _reports(["SG"])
    langs = list(config.COUNTRIES["SG"].drift_languages)
    n = len(pipeline.flagged_segments(inv, reps["SG"]))
    calls = []

    def fake(prompt, schema, **kw):
        calls.append(prompt)
        if len(calls) == 1:  # one bogus language and one duplicate are ignored
            return ([{"index": i, "language": lang.lower(), "text": f"{lang}{i}"}
                     for lang in langs for i in range(n)]
                    + [{"index": 0, "language": "Klingon", "text": "x"},
                       {"index": 0, "language": langs[0], "text": "dup"}])
        if len(calls) == 2:
            return [{"index": k, "text": f"BACK{k}"} for k in range(len(langs) * n)]
        return [{"index": k, "drift": "none", "explanation": "y"} for k in range(len(langs) * n)]

    monkeypatch.setattr(gemini, "generate_json", fake)
    out = pipeline.translation_check(inv, "SG", langs, mock=False)
    assert len(calls) == 3 and all(lang in calls[0] for lang in langs)
    assert len(out) == len(langs) * n
    assert [d.language for d in out] == [lang for lang in langs for _ in range(n)]
    assert "Klingon" not in calls[1] and "dup" not in calls[1]


def test_drift_without_report_orders_by_stage_a_leaves():
    inv, _ = _reports([])
    segs = pipeline.flagged_segments(inv)
    tagged = [s for s in segs if s.leaves]
    assert tagged and segs[: len(tagged)] == tagged
    checks, notes, errors = asyncio.run(pipeline.run_drift(inv, ["MY", "SG", "AU"], mock=True))
    assert not errors and not notes and set(checks) == {"MY", "SG"}


def test_drift_languages_in_hindi_my_malay():
    assert config.COUNTRIES["IN"].drift_languages == ("Hindi",)
    assert config.COUNTRIES["MY"].drift_languages == ("Malay",)


def test_drift_languages_sg_official_languages():
    assert config.COUNTRIES["SG"].drift_languages == ("Mandarin Chinese", "Malay", "Tamil")
    assert config.COUNTRIES["AU"].drift_languages == ()
