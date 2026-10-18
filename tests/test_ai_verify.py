"""Verification of stage A's signs of AI generation against frames (app/ai_verify.py)."""
import pytest

from app import ai_verify, gemini, synthetic
from app.ai_verify import Verdict
from app.schemas import Inventory, Segment, SyntheticCheck

from conftest import FFMPEG

# The live report of a real 30 s LTX clip: only the 0:19 cut artifact is really there.
LTX = ["0:05 the bottle label displays illegible, garbled text",
       "0:19 unnatural split-screen artifact and visual morphing during scene cut",
       "0:29 the subject's left hand has six fingers while waving"]
# A real Kling clip: watermark and garbled label are there; no time is named.
KLING = ["KlingAI watermark in the bottom right corner",
         "Garbled, non-alphabetic text on the bottle label",
         "Synthetic gibberish vocalization not matching human language",
         "Morphing details on the bottle and fingers"]


def _inv(signals, likelihood="high", duration=30.0, **kw) -> Inventory:
    return Inventory(duration_s=duration, segments=[
        Segment(start_s=0, end_s=11, transcript="hi", synthetic_signals=["garbled label text", "the voice is synthetic"]),
        Segment(start_s=11, end_s=duration, transcript="bye")],
        synthetic=SyntheticCheck(likelihood=likelihood, signals=signals, **kw))


def _answer(monkeypatch, visible: set[str], seen: list | None = None):
    """Fake `ask`: confirms the claims whose text is in `visible`."""
    def ask(video_path, claims):
        if seen is not None:
            seen.extend(claims)
        return {i: Verdict(index=i, visible=text in visible) for i, (text, _) in enumerate(claims)}
    monkeypatch.setattr(ai_verify, "ask", ask)


def test_times_and_audio_signs():
    assert ai_verify.times(LTX[0], 30) == [4.5, 5.0, 5.5]
    assert ai_verify.times("0:00 on-screen text", 30) == [0.0, 0.5]
    assert ai_verify.times("0:04-0:06 the face morphs", 30) == [4.0, 5.0, 6.0]
    assert ai_verify.times("0:29.8 hand", 30) == [29.3, 29.8, 29.95]  # clamped to the video
    assert ai_verify.times(KLING[0], 30) == []
    for s in ["Synthetic gibberish vocalization not matching human language",
              "the spoken audio uses a synthetic text-to-speech voice",
              "0:01 narrator states 'In today's AI-generated comedy sketch'",
              "the voice sounds synthetic or cloned (flat, metallic, no breaths)"]:
        assert ai_verify.is_audio(s), s
    for s in [*LTX, KLING[0], KLING[1], "lip movements do not match the speech",
              "0:00 on-screen text box displays 'AI-GENERATED SATIRE'"]:
        assert not ai_verify.is_audio(s), s


def test_invented_signs_are_dropped_and_likelihood_recomputed(monkeypatch):
    seen = []
    _answer(monkeypatch, {LTX[1]}, seen)
    inv = _inv(LTX)
    assert ai_verify.verify(None, inv, mock=False) is None
    syn = inv.synthetic
    assert syn.signals == [LTX[1]] and syn.unconfirmed == 2 and syn.verified
    assert syn.likelihood == "low"  # one sign left
    assert [ts for _, ts in seen] == [[4.5, 5.0, 5.5], [18.5, 19.0, 19.5], [28.5, 29.0, 29.5]]
    # stage A's per-moment signs are unverified: only the audio one stays
    assert inv.segments[0].synthetic_signals == ["the voice is synthetic"]
    # nothing is flagged on one sign: IN gets no AI finding
    assert synthetic.consequence("IN", syn, checked=True).outcome == "none"


def test_real_signs_are_kept(monkeypatch):
    seen = []
    _answer(monkeypatch, {KLING[0], KLING[1]}, seen)
    inv = _inv(KLING, label_present=True, label_text="KlingAI")
    ai_verify.verify(None, inv, mock=False)
    syn = inv.synthetic
    assert syn.signals == KLING[:3] and syn.unconfirmed == 1 and syn.likelihood == "high"
    # untimed signs are checked on frames spread over the video and where stage A saw them
    assert [text for text, _ in seen] == [KLING[0], KLING[1], KLING[3]]
    assert seen[0][1] == [3.75, 5.5, 11.25, 18.75, 26.25]
    assert not syn.voice_only


def test_audio_and_metadata_signs_are_not_sent(monkeypatch):
    seen = []
    _answer(monkeypatch, set(), seen)
    hint = 'file metadata: comment tag mentions "Veo"'
    inv = _inv(["the voice sounds synthetic", "0:03 the hand has six fingers", hint], likelihood="medium",
               metadata_signals=[hint])
    ai_verify.verify(None, inv, mock=False)
    assert [text for text, _ in seen] == ["0:03 the hand has six fingers"]
    syn = inv.synthetic
    assert syn.signals == ["the voice sounds synthetic", hint]
    assert syn.likelihood == "medium"  # a tool named in the file's tags is still a clear sign


def test_only_a_voice_left_is_voice_only(monkeypatch):
    _answer(monkeypatch, set())
    inv = _inv(["the voice sounds synthetic", "0:03 the face morphs"], likelihood="medium")
    ai_verify.verify(None, inv, mock=False)
    assert inv.synthetic.voice_only and inv.synthetic.likelihood == "low"
    assert synthetic.consequence("IN", inv.synthetic, checked=True).outcome == "none"


def test_failed_check_shows_no_visual_sign(monkeypatch):
    def broken(video_path, claims):
        raise RuntimeError("boom")
    monkeypatch.setattr(ai_verify, "ask", broken)
    inv = _inv([LTX[1], "the voice sounds synthetic"])
    error = ai_verify.verify(None, inv, mock=False)
    assert error and "could not be verified" in error
    assert inv.synthetic.signals == ["the voice sounds synthetic"] and not inv.synthetic.verified
    inv = _inv([LTX[1]])
    ai_verify.drop_visual(inv)  # e.g. the job's call budget was used up
    assert inv.synthetic.signals == [] and inv.synthetic.likelihood == "none"


def test_mock_and_nothing_to_check_change_nothing(monkeypatch):
    def never(*a):
        raise AssertionError("no call expected")
    monkeypatch.setattr(ai_verify, "ask", never)
    inv = _inv(LTX)
    assert ai_verify.verify(None, inv, mock=True) is None and inv.synthetic.signals == LTX
    inv = _inv(["the voice sounds synthetic"], likelihood="medium")
    ai_verify.verify(None, inv, mock=False)
    assert inv.synthetic.likelihood == "medium" and inv.synthetic.unconfirmed == 0


@pytest.mark.parametrize("likelihood, kept, meta, expected", [
    ("high", 0, False, "none"), ("high", 1, False, "low"), ("high", 2, False, "medium"),
    ("high", 3, False, "high"), ("medium", 3, False, "medium"), ("low", 2, False, "low"),
    ("high", 1, True, "medium"),
])
def test_recompute(likelihood, kept, meta, expected):
    signs = [f"s{i}" for i in range(kept)]
    assert ai_verify.recompute(likelihood, signs, ["m"] if meta else []) == expected


@pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg")
def test_one_call_with_every_frame(tiny_video, monkeypatch):
    path, real = tiny_video
    if not real:
        pytest.skip("needs a real video")
    sent = {}

    def fake(contents, schema, **kw):
        sent["contents"] = contents
        return [{"index": 0, "observed": "a hand with five fingers", "visible": False},
                {"index": 1, "observed": "a test pattern", "visible": True}]
    monkeypatch.setattr(gemini, "generate_json_with_images", fake)
    got = ai_verify.ask(path, [("0:01 six fingers", [0.5, 1.0]), ("odd colours", [1.0, 1.5])])
    assert {i: v.visible for i, v in got.items()} == {0: False, 1: True}
    images = [c for c in sent["contents"] if isinstance(c, bytes)]
    assert len(images) == 3 and all(b[:2] == b"\xff\xd8" for b in images)  # one JPEG per time
    assert '"frames": ["F1", "F2"]' in sent["contents"][-1]
