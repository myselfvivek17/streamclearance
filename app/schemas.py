"""Pydantic models shared by stage A, stage B, rule cards and the report."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

CountryCode = Literal["IN", "AU", "MY", "SG", "PK"]
AppliesTo = Literal["platform", "creator", "both"]
Risk = Literal["red", "amber", "green"]
RulesStatus = Literal["reviewed", "draft/unreviewed"]
Drift = Literal["none", "reduced", "increased"]
LawStatus = Literal["in_force", "bill", "amended", "guideline"]
DutyHolder = Literal["platform", "creator", "advertiser", "both"]
# content = says some kind of content is prohibited/harmful (a moment can be "covered by" it);
# obligation = what a platform must do about such content (duties, takedown, penalties).
ProvisionKind = Literal["content", "obligation"]
Badge = Literal["red", "amber", "green", "check"]

DISCLAIMER = (
    "Advisory only, not legal advice. Most of these laws place duties on platforms; "
    "the practical risk is takedown or geo-blocking in that country."
)
RISK_ORDER: dict[str, int] = {"green": 0, "amber": 1, "red": 2}


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
    categories: list[Category] = []  # legacy; derived from leaves when leaves are given

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
    note: str = ""


class CountryReport(BaseModel):
    country: CountryCode
    law: str
    label: str | None = None
    overall_risk: Risk
    findings: list[Finding] = []
    rules_status: RulesStatus = "reviewed"
    risk_framing: str = ""
    cited_cards: list[RuleCard] = []
    dropped_findings: int = 0
    uncovered: list[Uncovered] = []
    translation: list[DriftCheck] = []
    disclaimer: str = DISCLAIMER
    # Brain (SC_BRAIN=adk) only:
    brain: Literal["adk", "legacy"] = "legacy"
    laws: list[LawInfo] = []
    cited_provisions: list[Provision] = []
    platform_obligations: list[Provision] = []  # retrieved for the leaves found, not dumped
    critic_changes: int = 0
    notes: list[str] = []

    @computed_field
    @property
    def badge(self) -> Badge:
        """Grey "check" instead of green when sensitive moments are simply not covered."""
        if self.overall_risk == "green" and self.uncovered:
            return "check"
        return self.overall_risk


class Report(BaseModel):
    video_id: str
    duration_s: float = Field(ge=0)
    countries: list[CountryReport] = []
    mock: bool = False
    inventory: Inventory | None = None
    errors: list[str] = []
    disclaimer: str = DISCLAIMER
