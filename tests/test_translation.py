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


def test_only_flagged_segments_with_speech(monkeypatch):
    inv, reps = _reports(["IN"])
    segs = pipeline.flagged_segments(inv, reps["IN"])
    flagged_starts = {f.start_s for f in reps["IN"].findings}
    assert segs and all(s.start_s in flagged_starts and s.transcript for s in segs)
    monkeypatch.setattr(config, "MAX_DRIFT_SEGMENTS", 1)
    assert len(pipeline.flagged_segments(inv, reps["IN"])) == 1


def test_real_drift_translates_back_translates_blind_and_judges(monkeypatch):
    inv, reps = _reports(["PK"])
    calls = []

    def fake(prompt, schema, **kw):
        calls.append(prompt)
        n = len(pipeline.flagged_segments(inv, reps["PK"]))
        if len(calls) == 1:
            return [{"index": i, "text": f"URDU{i}"} for i in range(n)]
        if len(calls) == 2:
            return [{"index": i, "text": f"BACK{i}"} for i in range(n)]
        return [{"index": 0, "drift": "increased", "explanation": "x"}] + \
               [{"index": i, "drift": "none", "explanation": "y"} for i in range(1, n)]

    monkeypatch.setattr(gemini, "generate_json", fake)
    out = pipeline.translation_check(inv, reps["PK"], "Urdu", mock=False)
    assert len(calls) == 3
    assert "Urdu" in calls[0] and "fasting" in calls[0]
    assert "fasting" not in calls[1] and "URDU0" in calls[1]  # blind back-translation
    assert "BACK0" in calls[2] and "religious_offence" in calls[2]
    assert out[0].drift == "increased" and out[0].translated == "URDU0"
    assert out[0].back_translated == "BACK0"


def test_drift_languages_in_hindi_my_malay():
    assert config.COUNTRIES["IN"].drift_languages == ("Hindi",)
    assert config.COUNTRIES["MY"].drift_languages == ("Malay",)
