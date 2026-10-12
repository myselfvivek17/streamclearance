"""Pydantic models shared by stage A, stage B, rule cards and the report."""
from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

CountryCode = Literal["IN", "AU", "MY", "SG", "PK"]
AppliesTo = Literal["platform", "creator", "both"]
Risk = Literal["red", "amber", "green"]
# "not_checked": the country had zero verified provisions or rule cards to check against.
# Green means checked and nothing found; it is never used for a country that was not checked.
OverallRisk = Literal["red", "amber", "green", "not_checked"]
RulesStatus = Literal["reviewed", "draft/unreviewed"]
Drift = Literal["none", "reduced", "increased"]
LawStatus = Literal["in_force", "bill", "amended", "guideline"]
DutyHolder = Literal["platform", "creator", "advertiser", "both"]
# content = says some kind of content is prohibited/harmful (a moment can be "covered by" it);
# obligation = what a platform must do about such content (duties, takedown, penalties).
ProvisionKind = Literal["content", "obligation"]
Badge = Literal["red", "amber", "green", "check", "not_checked"]

DISCLAIMER = (
    "Advisory only, not legal advice. Most of these laws place duties on platforms; "
    "the practical risk is takedown or geo-blocking in that country."
)
REPORT_DISCLAIMER = "Automated pre-check, not legal advice. A person makes the publish decision."
RISK_ORDER: dict[str, int] = {"green": 0, "amber": 1, "red": 2}
NOT_CHECKED_LABEL = "Not checked: no verified law loaded"
NOT_CHECKED_FRAMING = ("Not checked: no verified law is loaded for this country, so nothing was "
                       "checked. This is not a clearance; check the video manually.")
UNCOVERED_NOTE = "The loaded law library does not cover this moment. Check it manually."
# Wording that reads like legal clearance. Never allowed in an uncovered / no-finding note:
# not being covered by the loaded laws says nothing about whether the creator is safe.
_CLEARANCE = re.compile(
    r"\b(?:no|not|nor|without|zero|free\s+(?:of|from))\b[\w\s,'-]{0,40}?\b(?:liab\w*|risk\w*|offen[cs]\w*|"
    r"violat\w*|breach\w*|illegal\w*|unlawful\w*|prohibit\w*|penalt\w*|issues?|problems?|concerns?)\b"
    r"|\b(?:safe\w*|clear(?:ed|ance|s)?|permitted|permissible|allowed|lawful|legal(?:ly)?|compliant|"
    r"complies|acceptable|fine|ok(?:ay)?|exempt\w*|harmless)\b",
    re.IGNORECASE)


def reads_like_clearance(text: str) -> bool:
    return bool(_CLEARANCE.search(text or ""))


def neutral_note(text: str) -> str:
    """The standard neutral wording, unless `text` is a neutral explanation (kept after it)."""
    text = (text or "").strip()
    if text.startswith(UNCOVERED_NOTE):
        text = text[len(UNCOVERED_NOTE):].strip()
    if not text or reads_like_clearance(text):
        return UNCOVERED_NOTE
    return f"{UNCOVERED_NOTE} {text}"


class Category(str, Enum):
    hate_or_incitement = "hate_or_incitement"
    religious_offence = "religious_offence"
    defamation_or_false_info = "defamation_or_false_info"
    sexual_or_nudity = "sexual_or_nudity"
    child_safety = "child_safety"
    violence_or_gore = "violence_or_gore"
    self_harm = "self_harm"
    terrorism_or_extremism = "terrorism_or_extremism"
    harassment_or_bullying = "harassment_or_bullying"
    privacy_or_doxxing = "privacy_or_doxxing"
    illegal_goods_or_gambling = "illegal_goods_or_gambling"
    misinformation_public_order = "misinformation_public_order"
    intellectual_property = "intellectual_property"
    national_security_or_sovereignty = "national_security_or_sovereignty"
    impersonation_or_fraud = "impersonation_or_fraud"
    other = "other"


class RuleCard(BaseModel):
    id: str
    country: CountryCode
    law: str
    section: str
    page: int = Field(ge=1)
    category: Category
    conduct: str
    applies_to: AppliesTo
    platform_duty: str
    penalty: str
    quote: str = Field(max_length=300, description="Verbatim text from the cited page")
    needs_review: bool = True


class Consequence(BaseModel):
    kind: str = "not specified"  # e.g. takedown, block, fine, prison, fine_or_prison
    timeframe: str = "not specified"
    penalty: str = "not specified"


class Review(BaseModel):
    status: Literal["draft", "reviewed"] = "draft"
    reviewer: str | None = None
    date: str | None = None


class Provision(BaseModel):
    """One legal rule (Layer 1 of the brain). Rule cards are a derived view of these."""
    id: str  # e.g. "MY/OSA2025/Sch1/para9"
    country: CountryCode
    law_id: str
    law_title: str
    law_status: LawStatus
    source_pdf: str
    page: int = Field(ge=1)
    section: str
    verbatim_text: str = Field(max_length=600, description="Verbatim text, verified on `page`")
    summary: str = ""
    targets: list[str] = Field(min_length=1, description="taxonomy leaves or parents")
    exceptions: list[str] = []
    duty_holder: DutyHolder
    consequence: Consequence = Consequence()
    review: Review = Review()
    kind: ProvisionKind = "content"


class LawInfo(BaseModel):
    """A law loaded for a country (shown on the country card)."""
    id: str
    title: str
    status: LawStatus
    label: str | None = None
    pdf_url: str
    provisions: int = 0
    source_sha256: str | None = None
    source_url: str | None = None  # the official file the law text was taken from
    source_page: str | None = None
    publisher: str | None = None
    version: str | None = None  # compilation / as-at / gazette date printed in the document
    source_verified: bool = False  # unverified laws are listed, never cited
    authoritative_language: str | None = None


SyntheticLikelihood = Literal["none", "low", "medium", "high"]


class SyntheticCheck(BaseModel):
    """Stage A: signs that the video was AI-generated or synthetically altered (video level).
    Evidence only: `signals` are observations, never a verdict."""
    likelihood: SyntheticLikelihood = "none"
    signals: list[str] = []
    label_present: bool = False  # the video itself discloses that it is AI-made
    label_text: str = ""
    depicts_real_person: bool = False  # an identifiable real public figure appears shown or voiced
    person_note: str = ""
    metadata_signals: list[str] = []  # AI-tool hints in the file's metadata tags (ffprobe), not the model


AIOutcome = Literal["none", "note", "flag", "check", "not_checked"]


class AIContentResult(BaseModel):
    """What the AI-content check means in one country (deterministic, app/synthetic.py)."""
    outcome: AIOutcome  # flag = a finding cites a provision; check = check manually; note = no flag
    risk: Risk | None = None  # set for flag (red/amber) and for an uncited amber
    leaf: str | None = None
    provision_ids: list[str] = []
    text: str = ""
    suggested_fix: str = ""


class _Timed(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self):
        if self.end_s < self.start_s:
            raise ValueError("end_s must be >= start_s")
        return self


class Segment(_Timed):
    """Stage A: one timestamped slice of the video."""
    transcript: str = ""
    language: str = ""
    claims: list[str] = []
    visuals: str = ""
    on_screen_text: str = ""
    leaves: list[str] = []  # taxonomy leaves (app/taxonomy.py)
    idioms: list[str] = []  # figures of speech that may turn literal in translation
    categories: list[Category] = []  # legacy; derived from leaves when leaves are given
    frame: str | None = None  # thumbnail file name (app/frames.py), GET /api/jobs/{id}/frames/{name}
    synthetic_signals: list[str] = []  # signs of AI generation seen in this segment

    @model_validator(mode="after")
    def _derive_categories(self):
        if self.leaves:
            from app import taxonomy

            self.leaves = taxonomy.clean(self.leaves)
            if self.leaves:
                self.categories = taxonomy.categories_for(self.leaves)
        return self


class Inventory(BaseModel):
    duration_s: float = Field(ge=0)
    languages: list[str] = []
    segments: list[Segment] = []
    speech_transcribed: bool | None = None  # None in mock mode
    synthetic: SyntheticCheck | None = None  # None in reports made before the AI-content check


class Finding(_Timed):
    """Stage B: one flagged moment checked against one country's rule cards."""
    category: Category
    rule_card_ids: list[str] = Field(min_length=1)
    risk: Risk
    suggested_fix: str
    rationale: str
    # Set by us from the cited cards, never by the model: True only when a cited
    # card says the poster (not just the platform) can be liable.
    creator_liability: bool = False
    # Brain (SC_BRAIN=adk) only:
    leaf: str | None = None
    exception_applies: bool = False
    exception_reason: str = ""
    critic_note: str = ""
    obligation_ids: list[str] = []  # platform-obligation provisions for this leaf
    video_level: bool = False  # about the whole video (AI-content check), not one moment


class DriftCheck(_Timed):
    """M4: does a flagged line's risk change once translated for this audience?"""
    language: str
    original: str
    translated: str
    back_translated: str
    drift: Drift
    explanation: str


class Uncovered(_Timed):
    """A sensitive moment (per stage A) that no provision of this country's law covers."""
    category: Category
    leaf: str | None = None
    note: str = UNCOVERED_NOTE
    video_level: bool = False  # about the whole video (AI-content check), not one moment

    @model_validator(mode="after")
    def _neutral(self):
        # Post-processing guard: the note comes from model output (assessor rationale, critic
        # reason) and must never state or imply that the creator is safe or not liable.
        self.note = neutral_note(self.note)
        return self


class CountryReport(BaseModel):
    country: CountryCode
    law: str
    label: str | None = None
    overall_risk: OverallRisk
    findings: list[Finding] = []
    rules_status: RulesStatus = "reviewed"
    risk_framing: str = ""
    cited_cards: list[RuleCard] = []
    dropped_findings: int = 0
    uncovered: list[Uncovered] = []
    translation: list[DriftCheck] = []
    translation_note: str = ""  # why lines were not checked, e.g. "No speech to translate."
    disclaimer: str = DISCLAIMER
    # Brain (SC_BRAIN=adk) only:
    brain: Literal["adk", "legacy"] = "legacy"
    laws: list[LawInfo] = []
    cited_provisions: list[Provision] = []
    platform_obligations: list[Provision] = []  # retrieved for the leaves found, not dumped
    critic_changes: int = 0
    notes: list[str] = []
    ai_content: AIContentResult | None = None  # the AI-content check's consequence here

    @model_validator(mode="after")
    def _no_clearance(self):
        if self.overall_risk == "not_checked":
            self.risk_framing = NOT_CHECKED_FRAMING
        elif not self.findings and reads_like_clearance(self.risk_framing):
            self.risk_framing = "No moment is covered by a loaded provision of this country's laws."
        return self

    @computed_field
    @property
    def badge(self) -> Badge:
        """Grey "check" instead of green when sensitive moments are simply not covered;
        at least amber when a line reads riskier once translated for this country;
        grey "not_checked" when there was no verified law to check against."""
        if self.overall_risk == "not_checked":
            return "amber" if any(d.drift == "increased" for d in self.translation) else "not_checked"
        if self.overall_risk != "red" and any(d.drift == "increased" for d in self.translation):
            return "amber"
        if self.overall_risk == "green" and self.uncovered:
            return "check"
        return self.overall_risk


class Privacy(BaseModel):
    """When the video was deleted (set once the job ends) and when the report will be."""
    video_deleted_at: str | None = None  # ISO 8601 UTC
    report_expires_at: str | None = None


class LawVersion(BaseModel):
    """One law a report was checked against, as it was at the time."""
    country: str
    law_id: str
    title: str
    legal_status: LawStatus | None = None  # in_force / bill / ...; None for basic coverage
    status: Literal["draft", "reviewed"]  # "reviewed" once a person signed off every provision
    source_file: str
    source_sha256: str | None = None  # of the law PDF the provisions were taken from
    provisions: int = 0
    provisions_built: str | None = None  # last commit of the provisions file (ISO 8601)
    source_url: str | None = None  # the official file the law text was taken from
    source_page: str | None = None
    publisher: str | None = None
    version: str | None = None  # compilation / as-at / gazette date printed in the document
    source_verified: bool = False  # unverified laws are listed, never cited
    authoritative_language: str | None = None


class ModelInfo(BaseModel):
    backend: str  # gemini-vertex, gemini-studio or mock
    fast: str
    smart: str
    location: str | None = None  # Vertex AI location


class Provenance(BaseModel):
    """How this report was made."""
    created_at: str  # ISO 8601 UTC
    app_git_sha: str
    brain: Literal["adk", "legacy"]
    model: ModelInfo
    laws: list[LawVersion] = []


class Report(BaseModel):
    video_id: str
    duration_s: float = Field(ge=0)
    countries: list[CountryReport] = []
    mock: bool = False
    inventory: Inventory | None = None
    errors: list[str] = []
    disclaimer: str = REPORT_DISCLAIMER
    privacy: Privacy | None = None
    provenance: Provenance | None = None
