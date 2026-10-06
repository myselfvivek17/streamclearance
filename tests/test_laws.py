import json

import pytest

from app import config, provisions
from app.provenance import file_hash


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


def test_registry_files_and_hashes():
    for law in provisions.registry():
        assert law.pdf_path.exists() and law.text_path.exists(), law.key
        if law.sha256:
            assert file_hash(law.pdf_path) == law.sha256, law.key
    recorded = [law for law in provisions.registry() if law.sha256]
    assert {law.key for law in recorded} >= {"AU/IGA2001", "AU/TGACODE2021", "SG/OSRA2025", "SG/GCA2022"}
    assert all(law.source_url and law.version for law in recorded)
