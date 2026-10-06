import importlib

import pytest

from app import config


@pytest.fixture
def reload_config(monkeypatch):
    yield lambda: importlib.reload(config)
    monkeypatch.delenv("SC_DATA_RESIDENCY", raising=False)
    importlib.reload(config)


def test_default_is_global_with_the_newest_model(reload_config, monkeypatch):
    for var in ("SC_DATA_RESIDENCY", "SC_MODEL_FAST", "SC_MODEL_SMART", "GCP_VERTEX_LOCATION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    c = reload_config()
    assert c.vertex_location() == "global" and c.SC_MODEL_FAST == c.SC_MODEL_SMART == "gemini-3.8-flash"


def test_sg_residency(reload_config, monkeypatch):
    for var in ("SC_MODEL_FAST", "SC_MODEL_SMART"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SC_DATA_RESIDENCY", "sg")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    c = reload_config()
    assert c.vertex_location() == "asia-southeast1"
    assert c.SC_MODEL_FAST == c.SC_MODEL_SMART == "gemini-2.5-flash"


def test_unknown_residency_rejected(reload_config, monkeypatch):
    monkeypatch.setenv("SC_DATA_RESIDENCY", "mars")
    with pytest.raises(ValueError):
        reload_config()
