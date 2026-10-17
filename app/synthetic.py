"""AI-generated content check: what stage A's signs of AI generation mean in each country.

Stage A reports evidence only (`Inventory.synthetic`: likelihood, observed signals,
whether the video carries an AI label, whether a real public figure seems to be shown
or voiced). This module turns that into a per-country consequence with a fixed table,
in code, never a model:

| signs (likelihood) | label | real person | consequence |
|---|---|---|---|
| none / low | any | any | note only |
| medium / high | yes | no | note only: the video discloses it |
| medium / high, voice only | no | no | check manually everywhere: a text-to-speech or synthetic-sounding voice alone is common narration, not a deceptive depiction |
| medium / high | no | no | IN: amber finding (labelling duty, IT Rules r3(3)(a)(ii), r4(1A)); elsewhere: check manually (no labelling law loaded) |
| medium / high | no | yes | red finding where a loaded law covers deepfakes / impersonation of a person (IN r3(3)(a)(i)(IV), r3(1)(b)(vi); SG OSRA s15); elsewhere amber, check manually |
| medium / high | yes | yes | amber, check manually: the label lowers the risk of deceiving viewers but does not settle it |

Countries with no verified law ("not checked") only get a note. Wording is always
"signs of AI generation", never "this is AI-generated".
"""
from __future__ import annotations

from app import config, events, provisions, taxonomy
from app.schemas import (
    RISK_ORDER, AIContentResult, CountryReport, Finding, Inventory, SyntheticCheck, Uncovered,
)

UNLABELLED = "misinfo.synthetic_unlabelled"
REAL_PERSON = "misinfo.deepfake_real_person"

# Provisions whose verbatim text genuinely covers each case (tests check they exist, are
# verified and quoted from the cited page). Only these are ever cited by this check.
PROVISIONS: dict[str, dict[str, list[str]]] = {
    "IN": {
        # labelling duty for synthetically generated information; uploaders must declare it
        UNLABELLED: ["IN/ITRULES2021/r3(3)(a)(ii)", "IN/ITRULES2021/r4(1A)"],
        # deceptive depiction of a natural person's identity, voice, conduct; impersonation
        REAL_PERSON: ["IN/ITRULES2021/r3(3)(a)(i)(IV)", "IN/ITRULES2021/r3(1)(b)(vi)"],
    },
    "SG": {
        REAL_PERSON: ["SG/OSRA2025/s15"],  # online impersonation
    },
}

FIX_LABEL = ("Add a clear “AI-generated” label on screen for the whole video (and say it at the "
             "start if it has audio), and declare it as AI-generated when you upload.")
FIX_VOICE = ("Optional: if the voice imitates a real person or the platform asks for AI "
             "disclosure, add an “AI voice” note in the caption or description.")
FIX_PERSON = ("Don’t publish synthetic footage or voice of a real person without their consent. If "
              "it is satire, make that unmistakable and label it as AI-generated.")


def _readable(pid: str) -> str:
    """"IN/ITRULES2021/r4(1A)" -> "IT Rules 2021 r4(1A)" for the activity feed."""
    from app.brain.tools import law_name
    parts = pid.split("/")
    return f"{law_name(parts[1])} {'/'.join(parts[2:])}" if len(parts) >= 3 else pid


_VOICE = ("voice", "voiceover", "voice-over", "narration", "narrator", "text-to-speech", "tts",
          "speech", "audio", "prosody", "intonation", "inflection", "breath")
_VISUAL = ("face", "hand", "finger", "lip", "blink", "frame", "lighting", "shadow", "morph",
           "artefact", "artifact", "image", "footage", "watermark", "background", "skin", "eye")


def voice_only(syn: SyntheticCheck) -> bool:
    """True when the only signs are a synthetic-sounding voice: the model says so, or every
    signal is about the voice and none about the picture. File-metadata hints never count."""
    if syn.metadata_signals or not syn.signals:
        return False
    if syn.voice_only:
        return True
    low = [s.lower() for s in syn.signals]
    return all(any(w in s for w in _VOICE) for s in low) and not any(w in s for s in low for w in _VISUAL)


def _signs(syn: SyntheticCheck) -> str:
    return "Signs of AI generation" + (": " + "; ".join(x.strip().rstrip(".") for x in syn.signals[:3]) if syn.signals else "")


def consequence(country: str, syn: SyntheticCheck, *, checked: bool) -> AIContentResult:
    name = config.COUNTRIES[country].name
    strong = syn.likelihood in ("medium", "high")
    if not checked:
        return AIContentResult(outcome="not_checked", text=(
            f"Not checked: no verified law is loaded for {name}." + (
                " Signs of AI generation were seen; check manually." if strong else "")))
    if not strong:
        return AIContentResult(outcome="none", text=(
            f"No clear signs of AI generation (likelihood {syn.likelihood}). This check only "
            "looks for visible and audible signs; it is not proof either way."))
    mapped = PROVISIONS.get(country, {})
    known = provisions.by_id(country)

    def cited(leaf: str) -> list[str]:
        ids = mapped.get(leaf, [])
        return ids if ids and all(pid in known for pid in ids) else []

    who = f" that appears to show or voice a real public figure ({syn.person_note})" \
        if syn.person_note else " that appears to show or voice a real public figure"
    if syn.depicts_real_person and syn.label_present:
        return AIContentResult(
            outcome="check", risk="amber", leaf=REAL_PERSON, suggested_fix=FIX_PERSON, text=(
                f"{_signs(syn)} in a video{who}. It carries an AI label "
                f"({events.quote(syn.label_text, 60)}), which lowers the risk of deceiving viewers "
                "but does not settle it. Check manually."))
    if syn.depicts_real_person:
        ids = cited(REAL_PERSON)
        if ids:
            return AIContentResult(
                outcome="flag", risk="red", leaf=REAL_PERSON, provision_ids=ids,
                suggested_fix=FIX_PERSON, text=(
                    f"{_signs(syn)} in a video{who}, with no AI label. {name}’s loaded law treats "
                    "deceptive depictions or impersonation of a real person as unlawful content."))
        return AIContentResult(
            outcome="check", risk="amber", leaf=REAL_PERSON, suggested_fix=FIX_PERSON, text=(
                f"{_signs(syn)} in a video{who}, with no AI label. No loaded law of {name} covers "
                "deepfakes or impersonation of a real person directly. Check manually: other laws "
                "(defamation, consumer, election) may apply."))
    if syn.label_present:
        return AIContentResult(outcome="note", text=(
            f"{_signs(syn)}. The video carries an AI label ({events.quote(syn.label_text, 60)}): "
            "noted, not flagged."))
    if voice_only(syn):
        return AIContentResult(
            outcome="check", leaf=UNLABELLED, suggested_fix=FIX_VOICE, text=(
                f"{_signs(syn)}, and no AI label. Only the voice seems synthetic (for example "
                "text-to-speech narration); no image or footage looked AI-made. A synthetic voice "
                "alone is common narration, so this is not flagged for "
                f"{name}. Check manually only if the voice imitates a real person."))
    ids = cited(UNLABELLED)
    if ids:
        return AIContentResult(
            outcome="flag", risk="amber", leaf=UNLABELLED, provision_ids=ids, suggested_fix=FIX_LABEL,
            text=(f"{_signs(syn)}, and no AI label. {name}’s IT Rules require synthetically generated "
                  "content to carry a prominent label, and platforms must ask uploaders to declare it."))
    return AIContentResult(
        outcome="check", leaf=UNLABELLED, suggested_fix=FIX_LABEL, text=(
            f"{_signs(syn)}, and no AI label. No loaded law of {name} has a labelling duty for "
            "AI-generated content; check the platforms’ own rules manually."))


def apply(rep: CountryReport, inventory: Inventory) -> None:
    """Attach the AI-content consequence to a country report (in place): a cited finding,
    or a "check manually" entry, never a silent change."""
    syn = inventory.synthetic
    if syn is None or rep.ai_content is not None:
        return
    res = consequence(rep.country, syn, checked=rep.overall_risk != "not_checked")
    rep.ai_content = res
    end = inventory.duration_s
    if res.outcome == "flag" and res.leaf and res.risk:
        by_id = provisions.by_id(rep.country)
        cited = [by_id[pid] for pid in res.provision_ids]
        rep.findings.append(Finding(
            start_s=0, end_s=end, category=taxonomy.category_of(res.leaf), rule_card_ids=res.provision_ids,
            risk=res.risk, suggested_fix=res.suggested_fix, rationale=res.text, leaf=res.leaf,
            creator_liability=any(p.duty_holder in ("creator", "advertiser", "both") for p in cited),
            video_level=True))
        have = {p.id for p in rep.cited_provisions}
        rep.cited_provisions += [p for p in cited if p.id not in have]
        have = {c.id for c in rep.cited_cards}
        rep.cited_cards += [provisions.to_card(p) for p in cited if p.id not in have]
    elif res.outcome == "check" and res.leaf:
        rep.uncovered.append(Uncovered(start_s=0, end_s=end, category=taxonomy.category_of(res.leaf),
                                       leaf=res.leaf, note=res.text, video_level=True))
    if res.risk and rep.overall_risk != "not_checked" \
            and RISK_ORDER[res.risk] > RISK_ORDER[rep.overall_risk]:
        rep.overall_risk = res.risk
    if res.outcome in ("flag", "check"):
        what = {"flag": f"{res.risk} finding citing {', '.join(_readable(x) for x in res.provision_ids)}",
                "check": f"check manually{' (amber)' if res.risk else ''}"}[res.outcome]
        events.emit("warn" if res.risk else "ai", f"AI-content check for {config.COUNTRIES[rep.country].name}: "
                    f"{what}", stage="ai", country=rep.country)
