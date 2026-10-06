"""How a report was made: law versions, model, app version (report `provenance`).

The app git sha and the date each provisions / rule-card file was last
committed come from git when it is there (local runs). The image has no .git,
so deploy.sh writes them into data/build_info.json first:
`python -m app.provenance` (gitignored). Without either, they are "unknown"
(dates fall back to the file's modification time).
"""
from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from app import config, provisions
from app.schemas import LawVersion, ModelInfo, Provenance

BUILD_INFO_PATH = config.DATA_DIR / "build_info.json"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def _rel(path: Path) -> str:
    path, root = path.resolve(), config.ROOT.resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def _built_files() -> list[Path]:
    """Files whose date is a law's "built" date: provisions and rule cards."""
    return sorted([*(config.DATA_DIR / "provisions").glob("*/*.json"), *config.RULES_DIR.glob("*.json")])


def collect_build_info() -> dict:
    """From git: the app sha (with -dirty for uncommitted changes) and file dates."""
    sha = _git("rev-parse", "--short", "HEAD")
    if sha and _git("status", "--porcelain", "--untracked-files=no", "--", "app", "data", "tools"):
        sha += "-dirty"
    dates = {}
    for path in _built_files():
        rel = _rel(path)
        if not _git("status", "--porcelain", "--", rel):  # committed as is
            date = _git("log", "-1", "--format=%cI", "--", rel)
            if date:
                dates[rel] = date
    return {"git_sha": sha, "file_dates": dates}


@functools.lru_cache(maxsize=1)
def build_info() -> dict:
    if BUILD_INFO_PATH.exists():
        return json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8"))
    return collect_build_info()


def _file_date(path: Path) -> str | None:
    if not path.exists():
        return None
    date = build_info().get("file_dates", {}).get(_rel(path))
    if date:
        return date
    ts = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


@functools.lru_cache(maxsize=64)
def _sha256(path: Path, mtime: float) -> str:  # mtime: a changed file is hashed again
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def file_hash(path: Path) -> str | None:
    return _sha256(path, path.stat().st_mtime) if path.exists() else None


def laws_used(country: str) -> list[LawVersion]:
    """The laws a check for this country reads: its provision files if it has
    any (deep coverage), else its rule-card file (basic coverage)."""
    provs = provisions.provisions_for(country)
    if provs:
        out = []
        for law in provisions.laws_for(country):
            mine = [p for p in provs if p.law_id == law.id]
            if not mine:
                continue
            reviewed = all(p.review.status == "reviewed" for p in mine)
            out.append(LawVersion(
                country=country, law_id=law.id, title=law.title, legal_status=law.status,
                status="reviewed" if reviewed else "draft", source_file=law.pdf,
                source_sha256=file_hash(law.pdf_path), provisions=len(mine),
                provisions_built=_file_date(law.provisions_path)))
        return out
    c = config.COUNTRIES[country]
    from app.pipeline import load_cards

    cards, status = load_cards(country)
    cards_path = c.rules_path if c.rules_path.exists() else c.dryrun_rules_path
    return [LawVersion(
        country=country, law_id=c.source, title=c.law, legal_status=None,
        status="reviewed" if status == "reviewed" else "draft", source_file=c.pdf_path.name,
        source_sha256=file_hash(c.pdf_path), provisions=len(cards),
        provisions_built=_file_date(cards_path))]


def model_info(mock: bool) -> ModelInfo:
    if mock:
        return ModelInfo(backend="mock", fast="mock", smart="mock", location=None)
    backend = config.gemini_backend()
    return ModelInfo(backend=f"gemini-{backend}", fast=config.SC_MODEL_FAST, smart=config.SC_MODEL_SMART,
                     location=config.vertex_location() if backend == "vertex" else None)


def provenance(countries: list[str], mock: bool, created_at: float) -> Provenance:
    return Provenance(
        created_at=datetime.fromtimestamp(created_at, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        app_git_sha=build_info().get("git_sha") or "unknown",
        brain=config.brain(),
        model=model_info(mock),
        laws=[law for cc in countries for law in laws_used(cc)],
    )


if __name__ == "__main__":  # deploy.sh: python -m app.provenance
    info = collect_build_info()
    BUILD_INFO_PATH.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(f"{_rel(BUILD_INFO_PATH)}: {info['git_sha']}, {len(info['file_dates'])} file dates", file=sys.stderr)
