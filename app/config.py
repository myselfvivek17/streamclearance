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

# SC_DATA_RESIDENCY=sg keeps Gemini processing in Singapore: Vertex AI location
# asia-southeast1 with gemini-2.5-flash, because gemini-3.8-flash is not served
# there (404). Unset (the default): location `global` (Google picks the region)
# with gemini-3.8-flash.
RESIDENCY = {"sg": ("asia-southeast1", "gemini-2.5-flash")}
DATA_RESIDENCY = os.getenv("SC_DATA_RESIDENCY", "").strip().lower() or None
if DATA_RESIDENCY is not None and DATA_RESIDENCY not in RESIDENCY:
    raise ValueError(f"SC_DATA_RESIDENCY must be one of {', '.join(RESIDENCY)} or unset")
_DEFAULT_MODEL = RESIDENCY[DATA_RESIDENCY][1] if DATA_RESIDENCY else "gemini-3.8-flash"

SC_MODEL_FAST = os.getenv("SC_MODEL_FAST", _DEFAULT_MODEL)
SC_MODEL_SMART = os.getenv("SC_MODEL_SMART", _DEFAULT_MODEL)

# Local storage root (uploads/ and jobs/ live under it). Swap app/storage.py for GCS later.
STORAGE_DIR = Path(os.getenv("SC_STORAGE_DIR", str(ROOT)))

MAX_UPLOAD_BYTES = int(os.getenv("SC_MAX_UPLOAD_MB", "200")) * 1024 * 1024
MAX_DURATION_S = float(os.getenv("SC_MAX_DURATION_S", "600"))
VIDEO_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}

# At most this many Gemini calls in flight per job.
GEMINI_CONCURRENCY = int(os.getenv("SC_GEMINI_CONCURRENCY", "2"))
GEMINI_MAX_RETRIES = int(os.getenv("SC_GEMINI_MAX_RETRIES", "5"))
# Cap translation-drift checks so a long video can't burn the quota.
MAX_DRIFT_SEGMENTS = int(os.getenv("SC_MAX_DRIFT_SEGMENTS", "6"))


def gemini_api_key() -> str | None:
    """AI Studio key. Read lazily so it is never captured at import time."""
    return os.getenv("GEMINI_API_KEY") or None


def gcp_project() -> str | None:
    return os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or None


def vertex_location() -> str:
    if DATA_RESIDENCY:  # wins over GOOGLE_CLOUD_LOCATION (deploy.sh sets global)
        return RESIDENCY[DATA_RESIDENCY][0]
    return os.getenv("GCP_VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "global"


def gemini_backend() -> str | None:
    """"vertex" (Vertex AI, ADC / service account), "studio" (AI Studio key) or None.

    Vertex wins when SC_GEMINI_BACKEND=vertex or a GCP project is set.
    """
    forced = os.getenv("SC_GEMINI_BACKEND", "").strip().lower()
    if forced not in ("", "vertex", "studio"):
        raise ValueError("SC_GEMINI_BACKEND must be vertex or studio")
    if forced == "vertex" or (not forced and gcp_project()):
        return "vertex"
    if gemini_api_key():
        return "studio"
    return None


def apply_adk_env() -> None:
    """ADK builds its own genai client from env vars; point it at the same backend."""
    if gemini_backend() == "vertex":
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
        if gcp_project():
            os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project()
        os.environ["GOOGLE_CLOUD_LOCATION"] = vertex_location()
    elif gemini_backend() == "studio":
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"


# Inline video bytes up to this size on Vertex; larger files go through GCS.
VERTEX_INLINE_MAX_BYTES = int(float(os.getenv("SC_VERTEX_INLINE_MB", "19")) * 1024 * 1024)


def upload_bucket() -> str:
    """GCS bucket for videos too large to inline (lifecycle: deleted after 1 day)."""
    return os.getenv("SC_UPLOAD_BUCKET") or f"{gcp_project()}-sc-uploads"


# Law search (retrieval path B): Vertex AI RAG Engine, one corpus per country
# named <SC_SEARCH_STORE>-<cc>; tools/index_laws.py syncs it and records the
# uploaded documents in data/search_index.json.
SEARCH_STORE = os.getenv("SC_SEARCH_STORE", "streamclearance-laws")
SEARCH_INDEX_PATH = DATA_DIR / "search_index.json"


def search_location() -> str:
    """RAG Engine region (preview in asia-southeast1, close to Cloud Run)."""
    return os.getenv("SC_SEARCH_LOCATION", "asia-southeast1")


def search_max_distance() -> float:
    """Cosine-distance cutoff for law-search hits. The service default (~0.5) drops
    relevant hits: measured with text-embedding-005, on-topic hits score 0.47-0.55
    and unrelated queries 0.60+."""
    return float(os.getenv("SC_SEARCH_MAX_DISTANCE", "0.6"))


def law_search() -> str:
    """"rag" (Vertex AI RAG Engine) or "bm25" (local keyword search).

    SC_LAW_SEARCH overrides; by default rag on the Vertex backend, else bm25.
    """
    forced = os.getenv("SC_LAW_SEARCH", "").strip().lower()
    if forced not in ("", "rag", "bm25"):
        raise ValueError("SC_LAW_SEARCH must be rag or bm25")
    if forced:
        return forced
    return "rag" if provider() == "gemini" and gemini_backend() == "vertex" else "bm25"


# Cost guard: hard cap on model calls (direct + ADK) per analysis job.
MAX_CALLS_PER_JOB = int(os.getenv("SC_MAX_CALLS_PER_JOB", "40"))


def provider() -> str:
    """SC_MOCK=1 forces mock; else "gemini" (Vertex or AI Studio) if configured; else mock."""
    if os.getenv("SC_MOCK") == "1":
        return "mock"
    return "gemini" if gemini_backend() else "mock"


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


def unused_video_ttl_s() -> float:
    """An uploaded video that is never analysed is deleted after this long."""
    return float(os.getenv("SC_UNUSED_VIDEO_TTL_MIN", "60")) * 60


def report_ttl_days() -> float:
    """Reports (job records) are deleted this many days after the job was created."""
    return float(os.getenv("SC_REPORT_TTL_DAYS", "7"))


def app_region() -> str | None:
    """Cloud Run region the app (and its local storage) runs in; deploy.sh sets it."""
    return os.getenv("SC_APP_REGION") or None


def vertex_cache_disabled() -> bool:
    """Set SC_VERTEX_CACHE_DISABLED=1 once the project's Vertex AI cacheConfig has
    disableCache: true (README "Manual step for zero data retention"). Only
    changes what the privacy page says."""
    return os.getenv("SC_VERTEX_CACHE_DISABLED") == "1"


def no_training() -> bool:
    """Vertex AI never trains on customer data (Service Specific Terms); mock sends
    nothing anywhere. AI Studio's free tier may, so no such claim there."""
    return provider() == "mock" or gemini_backend() == "vertex"


def access_code() -> str | None:
    return os.getenv("SC_ACCESS_CODE") or None


def ffprobe_path() -> str | None:
    found = shutil.which("ffprobe")
    if found:
        return found
    bin_dir = os.getenv("SC_FFMPEG_DIR")
    candidate = Path(bin_dir) / "ffprobe" if bin_dir else None
    return str(candidate) if candidate and candidate.exists() else None


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
            "Online Safety (Relief and Accountability) Act 2025 (No. 23 of 2025)",
            "singapore_online_safety_relief_accountability_act_2025",
            drift_languages=("Mandarin Chinese", "Malay", "Tamil"),
        ),
        CountryLaw(
            "PK", "Pakistan",
            "Prevention of Electronic Crimes (Amendment) Act, 2025",
            "pakistan_peca_amendment_2025",
            drift_languages=("Urdu",),
        ),
    ]
}
