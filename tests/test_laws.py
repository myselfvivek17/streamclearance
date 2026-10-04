import json

import pytest

from app import config


@pytest.mark.parametrize("code", sorted(config.COUNTRIES))
def test_law_files_present(code):
    c = config.COUNTRIES[code]
    assert c.pdf_path.exists()
    pages = json.loads(c.text_path.read_text())
    assert [p["page"] for p in pages] == list(range(1, len(pages) + 1))
    assert sum(len(p["text"]) for p in pages) > 10_000


def test_singapore_is_labelled_as_bill():
    assert "verify enacted Act" in config.COUNTRIES["SG"].label
