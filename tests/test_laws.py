import json
import re
from urllib.parse import urlparse

import pytest

from app import config, provisions
from app.provenance import file_hash
from tools.build_rules import quote_on_page


@pytest.mark.parametrize("code", sorted(config.COUNTRIES))
def test_law_files_present(code):
    c = config.COUNTRIES[code]
    assert c.pdf_path.exists()
    pages = json.loads(c.text_path.read_text())
    assert [p["page"] for p in pages] == list(range(1, len(pages) + 1))
    assert sum(len(p["text"]) for p in pages) > 10_000


def test_singapore_uses_the_enacted_act():
    sg = config.COUNTRIES["SG"]
    assert "Act 2025" in sg.law and "Bill" not in sg.law and sg.label is None
    assert all(law.status != "bill" for law in provisions.laws_for("SG"))


REQUIRED = ("source_url", "publisher", "version", "retrieved_at", "sha256", "language",
            "authoritative_language")
OFFICIAL_HOSTS = ("legislation.gov.au", "sso.agc.gov.sg", "lom.agc.gov.my", ".gov.my", ".gov.in",
                  ".nic.in", ".gov.pk", ".gov.sg")


def test_registry_files_and_hashes():
    for law in provisions.registry():
        assert law.pdf_path.exists() and law.text_path.exists(), law.key
        assert file_hash(law.pdf_path) == law.sha256, law.key


def test_every_raw_file_is_in_the_registry():
    listed = {law.pdf for law in provisions.registry()}
    assert {p.name for p in config.LAWS_RAW_DIR.glob("*.pdf")} == listed


@pytest.mark.parametrize("law", provisions.registry(), ids=lambda law: law.key)
def test_registry_source_fields(law):
    for field in REQUIRED:
        assert getattr(law, field), f"{law.key}: {field}"
    assert isinstance(law.source_verified, bool)
    assert law.language == "en"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", law.retrieved_at)
    host = urlparse(law.source_url).hostname or ""
    assert law.source_url.startswith("https://") and host.endswith(OFFICIAL_HOSTS), law.key
    assert urlparse(law.source_url).path not in ("", "/"), f"{law.key}: a file URL, not a home page"
    if law.source_page:
        assert (urlparse(law.source_page).hostname or "").endswith(OFFICIAL_HOSTS), law.key
    if not law.source_verified:
        assert law.text_source or law.unverified_reason, f"{law.key}: say why it is unverified"


def test_unverified_laws_load_no_provisions(real_provisions):
    unverified = {law.key for law in provisions.registry() if not law.source_verified}
    assert unverified == {"IN/DMR1954", "PK/PECA2025"}
    loaded = {f"{p.country}/{p.law_id}" for provs in provisions.all_provisions().values() for p in provs}
    assert loaded and not loaded & unverified
    infos = {i.id: i for i in provisions.law_infos("IN")}
    assert infos["DMR1954"].provisions == 0 and not infos["DMR1954"].source_verified


@pytest.mark.parametrize("law", [law for law in provisions.registry() if law.provisions_path.exists()],
                         ids=lambda law: law.key)
def test_every_provision_quote_is_on_its_page(law, real_provisions):
    pages = {p["page"]: p["text"] for p in json.loads(law.text_path.read_text(encoding="utf-8"))}
    for p in json.loads(law.provisions_path.read_text(encoding="utf-8")):
        assert p["source_pdf"] == law.pdf, p["id"]
        assert quote_on_page(p["verbatim_text"], pages.get(p["page"], "")), p["id"]
