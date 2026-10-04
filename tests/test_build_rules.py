import json

import pytest

from app import config, gemini
from app.schemas import RuleCard
from tools import build_rules
from tools.build_rules import load_pages, quote_on_page

INDIA = config.COUNTRIES["IN"]


@pytest.fixture(scope="module")
def india_pages():
    return load_pages(INDIA)


def test_real_quote_accepted(india_pages):
    # Rule 3(1)(b)(ii), page 6 of the real India text; line breaks and curly
    # apostrophe differ from the extracted text.
    quote = ("is obscene, pornographic, paedophilic, invasive of another's privacy "
             "including bodily privacy, insulting or harassing on the basis of gender")
    assert quote_on_page(quote, india_pages[6])
    assert quote_on_page(quote.upper(), india_pages[6])


def test_quote_survives_pypdf_split_words(india_pages):
    # pypdf yields "Constituti on" on page 5; the real word must still match.
    assert "Constituti on" in india_pages[5]
    assert quote_on_page("specified in the Eighth Schedule to the Constitution", india_pages[5])


def test_invented_quote_rejected(india_pages):
    assert not quote_on_page("creators shall be imprisoned for any video that offends anyone",
                             india_pages[6])
    # real text, wrong page
    assert not quote_on_page("is obscene, pornographic, paedophilic, invasive", india_pages[1])
    assert not quote_on_page("   ", india_pages[6])


def test_dry_run_writes_valid_file(tmp_path):
    assert build_rules.main(["--country", "IN", "--dry-run", "--out-dir", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "in.json").read_text())
    assert data, "expected at least one card"
    cards = [RuleCard.model_validate(c) for c in data]
    pages = load_pages(INDIA)
    assert len({c.id for c in cards}) == len(cards)
    for c in cards:
        assert c.country == "IN" and c.section and c.page in pages and c.needs_review
        assert quote_on_page(c.quote, pages[c.page])
    # the mock's invented quotes must all have been dropped
    assert not any("imprisoned" in c.quote for c in cards)


def test_bad_page_and_long_quote_rejected(india_pages):
    real = "is obscene, pornographic, paedophilic"
    base = dict(section="s", category="other", conduct="c", applies_to="platform",
                platform_duty="d", penalty="p")
    fake = lambda prompt: [  # noqa: E731
        dict(base, page=6, quote=real),
        dict(base, page=6, quote=real),            # duplicate
        dict(base, page=999, quote=real),          # page missing
        dict(base, page=6, quote=real + " x" * 200),  # too long
        dict(base, page=6, category="bogus", quote=real),  # bad enum
    ]
    cards, rejected = build_rules.build_cards(INDIA, fake)
    n_chunks = len(build_rules.chunk_pages(india_pages))
    assert len(cards) == 1
    assert len(rejected) == 5 * n_chunks - 1


def test_chunks_keep_every_page_once(india_pages):
    chunks = build_rules.chunk_pages(india_pages)
    assert [p for c in chunks for p in c] == sorted(india_pages)
    assert all(len(c) <= build_rules.MAX_PAGES_PER_CHUNK for c in chunks)


def test_real_path_goes_through_gemini_module(monkeypatch):
    calls = []

    def fake_generate_json(prompt, schema, **kw):
        calls.append((prompt, kw))
        return []

    monkeypatch.setattr(gemini, "generate_json", fake_generate_json)
    cards, rejected = build_rules.build_cards(INDIA, build_rules.gemini_generate)
    assert cards == [] and rejected == []
    assert calls and "=== PAGE 1 ===" in calls[0][0]
    assert "PLATFORMS" in calls[0][1]["system_instruction"]


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini, "_client", None)
    with pytest.raises(gemini.GeminiUnavailable):
        gemini.generate_json("hi", dict)
