"""Central configuration. Values are env-overridable; .env is loaded if present."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
LAWS_RAW_DIR = DATA_DIR / "laws" / "raw"
LAWS_TEXT_DIR = DATA_DIR / "laws" / "text"
RULES_DIR = DATA_DIR / "rules"

SC_MODEL_FAST = os.getenv("SC_MODEL_FAST", "gemini-3.8-flash")
SC_MODEL_SMART = os.getenv("SC_MODEL_SMART", "gemini-3.8-flash")


def gemini_api_key() -> str | None:
    """AI Studio key (not Vertex). Read lazily so it is never captured at import time."""
    return os.getenv("GEMINI_API_KEY") or None


@dataclass(frozen=True)
class CountryLaw:
    code: str
    name: str
    law: str
    source: str  # file stem in data/laws/raw and data/laws/text
    label: str | None = None  # caveat shown alongside citations

    @property
    def pdf_path(self) -> Path:
        return LAWS_RAW_DIR / f"{self.source}.pdf"

    @property
    def text_path(self) -> Path:
        return LAWS_TEXT_DIR / f"{self.source}.json"

    @property
    def rules_path(self) -> Path:
        return RULES_DIR / f"{self.code.lower()}.json"


COUNTRIES: dict[str, CountryLaw] = {
    c.code: c
    for c in [
        CountryLaw(
            "IN", "India",
            "Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Rules, 2021 (as updated 10.02.2026)",
            "india_it_rules_2021_feb2026",
        ),
        CountryLaw(
            "AU", "Australia",
            "Online Safety Act 2021 (Compilation, 11 Dec 2024)",
            "australia_online_safety_act_2021_comp2024-12-11",
        ),
        CountryLaw(
            "MY", "Malaysia",
            "Online Safety Act 2025 (Act 866)",
            "malaysia_online_safety_act_2025",
        ),
        CountryLaw(
            "SG", "Singapore",
            "Online Safety (Relief and Accountability) Bill 2025",
            "singapore_online_safety_bill_2025",
            label="Bill as introduced 2025 — verify enacted Act",
        ),
        CountryLaw(
            "PK", "Pakistan",
            "Prevention of Electronic Crimes (Amendment) Act, 2025",
            "pakistan_peca_amendment_2025",
        ),
    ]
}
