"""Pydantic models shared by stage A, stage B, rule cards and the report."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

CountryCode = Literal["IN", "AU", "MY", "SG", "PK"]
AppliesTo = Literal["platform", "creator", "both"]
Risk = Literal["red", "amber", "green"]


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
    categories: list[Category] = []


class Inventory(BaseModel):
    duration_s: float = Field(ge=0)
    languages: list[str] = []
    segments: list[Segment] = []


class Finding(_Timed):
    """Stage B: one flagged moment checked against one country's rule cards."""
    category: Category
    rule_card_ids: list[str] = Field(min_length=1)
    risk: Risk
    suggested_fix: str
    rationale: str


class CountryReport(BaseModel):
    country: CountryCode
    law: str
    label: str | None = None
    overall_risk: Risk
    findings: list[Finding] = []


class Report(BaseModel):
    video_id: str
    duration_s: float = Field(ge=0)
    countries: list[CountryReport] = []
    disclaimer: str = (
        "Advisory only, not legal advice. Most of these laws place duties on platforms; "
        "the practical risk is takedown or geo-blocking in that country."
    )
