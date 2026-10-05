import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app import config, provisions, storage


def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    bin_dir = os.getenv("SC_FFMPEG_DIR")
    candidate = Path(bin_dir) / "ffmpeg" if bin_dir else None
    return str(candidate) if candidate and candidate.exists() else None


FFMPEG = _ffmpeg()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Mock mode, no delays, no access code, storage and generated rules in tmp."""
    monkeypatch.setenv("SC_MOCK", "1")
    monkeypatch.setenv("SC_MOCK_DELAY", "0")
    monkeypatch.delenv("SC_ACCESS_CODE", raising=False)
    for var in ("GEMINI_API_KEY", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "SC_GEMINI_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(storage, "_storage", storage.LocalStorage(tmp_path / "store"))
    rules = tmp_path / "rules"
    rules.mkdir()
    monkeypatch.setattr(config, "RULES_DIR", rules)
    # No provisions unless a test asks for them (see the `real_provisions` fixture).
    monkeypatch.setattr(provisions, "PROVISIONS_DIR", tmp_path / "provisions")
    provisions.reload()
    yield
    provisions.reload()


@pytest.fixture
def real_provisions(monkeypatch):
    """The provisions committed under data/provisions."""
    monkeypatch.setattr(provisions, "PROVISIONS_DIR", config.DATA_DIR / "provisions")
    provisions.reload()


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory) -> tuple[Path, bool]:
    """A 2-second mp4 (ffmpeg) or a dummy file. Returns (path, is_real_video)."""
    out = tmp_path_factory.mktemp("video") / "tiny.mp4"
    if FFMPEG and config.ffprobe_path():
        subprocess.run(
            [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
             "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
            check=True, timeout=60,
        )
        return out, True
    out.write_bytes(b"\x00" * 2048)
    return out, False
