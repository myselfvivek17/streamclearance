"""Central configuration. Values are env-overridable; .env is loaded if present."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
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

# Local storage root (uploads/ and jobs/ live under it). Swap app/storage.py for GCS later.
STORAGE_DIR = Path(os.getenv("SC_STORAGE_DIR", str(ROOT)))

MAX_UPLOAD_BYTES = int(os.getenv("SC_MAX_UPLOAD_MB", "200")) * 1024 * 1024
MAX_DURATION_S = float(os.getenv("SC_MAX_DURATION_S", "600"))
VIDEO_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}

# Free-tier friendly: at most this many Gemini calls in flight per job.
GEMINI_CONCURRENCY = int(os.getenv("SC_GEMINI_CONCURRENCY", "2"))
GEMINI_MAX_RETRIES = int(os.getenv("SC_GEMINI_MAX_RETRIES", "5"))
# Cap translation-drift checks so a long video can't burn the quota.
MAX_DRIFT_SEGMENTS = int(os.getenv("SC_MAX_DRIFT_SEGMENTS", "6"))


def gemini_api_key() -> str | None:
    """AI Studio key (not Vertex). Read lazily so it is never captured at import time."""
    return os.getenv("GEMINI_API_KEY") or None


PROVIDERS = ("gemini", "openrouter", "mock")


def provider() -> str:
    """SC_MOCK=1 forces mock; else SC_PROVIDER if set; else the first configured key; else mock."""
    if os.getenv("SC_MOCK") == "1":
        return "mock"
    forced = os.getenv("SC_PROVIDER", "").strip().lower()
    if forced:
        if forced not in PROVIDERS:
            raise ValueError(f"SC_PROVIDER must be one of {', '.join(PROVIDERS)}")
        return forced
    if gemini_api_key():
        return "gemini"
    if os.getenv("OPENROUTER_API_KEY"): return "openrouter"  # OPENROUTER-TEMP
    return "mock"


if provider() == "openrouter" and not os.getenv("SC_GEMINI_CONCURRENCY"): GEMINI_CONCURRENCY = 1  # OPENROUTER-TEMP


def mock_mode() -> bool:
    """Mock unless a provider key is set; SC_MOCK=1 forces mock even with a key."""
    return provider() == "mock"


def brain() -> str:
    """SC_BRAIN=adk (default): ADK Workflow over provisions; "legacy": stage A + stage B on rule cards."""
    value = os.getenv("SC_BRAIN", "adk").strip().lower()
    if value not in ("adk", "legacy"):
        raise ValueError("SC_BRAIN must be adk or legacy")
    return value


def mock_delay_s() -> float:
    """Pause per mocked stage so the progress UI is visible (0 in tests)."""
    return float(os.getenv("SC_MOCK_DELAY", "0.6"))


def access_code() -> str | None:
    return os.getenv("SC_ACCESS_CODE") or None


def ffprobe_path() -> str | None:
    found = shutil.which("ffprobe")
    if found:
        return found
    bundled = sorted(Path.home().glob(".local/tools/ffmpeg-*/bin/ffprobe"))
    return str(bundled[-1]) if bundled else None


@dataclass(frozen=True)
class CountryLaw:
    code: str
    name: str
    law: str
    source: str  # file stem in data/laws/raw and data/laws/text
    label: str | None = None  # caveat shown alongside citations
    # Languages for the translation-drift check (English-speaking audiences need none).
    drift_languages: tuple[str, ...] = field(default=())

    @property
    def pdf_path(self) -> Path:
        return LAWS_RAW_DIR / f"{self.source}.pdf"

    @property
    def text_path(self) -> Path:
        return LAWS_TEXT_DIR / f"{self.source}.json"

    @property
    def rules_path(self) -> Path:
        return RULES_DIR / f"{self.code.lower()}.json"

    @property
    def dryrun_rules_path(self) -> Path:
        return RULES_DIR / f"{self.code.lower()}.dryrun.json"


COUNTRIES: dict[str, CountryLaw] = {
    c.code: c
    for c in [
        CountryLaw(
            "IN", "India",
            "Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Rules, 2021 (as updated 10.02.2026)",
            "india_it_rules_2021_feb2026",
            drift_languages=("Hindi",),
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
            drift_languages=("Malay",),
        ),
        CountryLaw(
            "SG", "Singapore",
            "Online Safety (Relief and Accountability) Bill 2025",
            "singapore_online_safety_bill_2025",
            label="Bill as introduced 2025 — verify enacted Act",
            drift_languages=("Mandarin Chinese",),
        ),
        CountryLaw(
            "PK", "Pakistan",
            "Prevention of Electronic Crimes (Amendment) Act, 2025",
            "pakistan_peca_amendment_2025",
            drift_languages=("Urdu",),
        ),
    ]
}
