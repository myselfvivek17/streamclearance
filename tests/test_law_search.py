"""Path B on Vertex AI RAG Engine: hit mapping, fallback and the indexer (no network)."""
import asyncio
from types import SimpleNamespace

import pytest

from app import config, law_search, provisions, retrieval
from app.brain.tools import make_tools
from app.law_search import Context
from app.schemas import Provision
from tools import index_laws

CORPUS = "projects/1/locations/asia-southeast1/ragCorpora/7"
MAS_TITLE = "Medicines (Advertisement and Sale) Act 1956 (Act 290)"  # from the registry


def _prov(pid, law, page, *, kind="content", text="", targets=("health.cure_claim",)):
    return Provision(
        id=f"MY/{law}/{pid}", country="MY", law_id=law, law_title=f"{law} title", law_status="in_force",
        source_pdf="x.pdf", page=page, section=pid, verbatim_text=text or f"text {pid}",
        targets=list(targets), duty_holder="platform", kind=kind)


@pytest.fixture
def rag(monkeypatch):
    """Two MY provisions, a fake index and a fake RAG Engine returning `calls['contexts']`."""
    data = {"MY": [
        _prov("s3", "MAS1956", 5, text="advertisement for the prevention or treatment of diabetes"),
        _prov("s13", "OSA2025", 9, kind="obligation", text="licensed platform shall mitigate harmful content",
              targets=("harm",)),
    ]}
    monkeypatch.setattr(provisions, "_cache", data)
    names = {p.id: index_laws.provision_doc_name(p) for p in data["MY"]}
    index = {"location": "asia-southeast1", "corpora": {"MY": CORPUS}, "docs": {
        names["MY/MAS1956/s3"]: {"country": "MY", "doc_type": "provision", "provision_id": "MY/MAS1956/s3"},
        names["MY/OSA2025/s13"]: {"country": "MY", "doc_type": "provision", "provision_id": "MY/OSA2025/s13"},
        "t_MY_MAS1956_p005.txt": {"country": "MY", "doc_type": "full_text", "law_id": "MAS1956", "page": 5},
        "t_MY_MAS1956_p011.txt": {"country": "MY", "doc_type": "full_text", "law_id": "MAS1956", "page": 11},
        "t_IN_DMR1954_p002.txt": {"country": "IN", "doc_type": "full_text", "law_id": "DMR1954", "page": 2},
    }}
    monkeypatch.setattr(law_search, "_index", index)
    calls = {"n": 0, "contexts": []}

    def fake_retrieve(corpus, query, top_k, *, location=None):
        calls["n"] += 1
        calls["args"] = (corpus, query, top_k, location)
        if isinstance(calls["contexts"], Exception):
            raise calls["contexts"]
        return calls["contexts"]

    monkeypatch.setattr(law_search, "retrieve", fake_retrieve)
    monkeypatch.setenv("SC_LAW_SEARCH", "rag")
    retrieval.reset()
    yield SimpleNamespace(calls=calls, names=names)
    retrieval.reset()
    monkeypatch.setattr(provisions, "_cache", None)


def test_law_search_selection(monkeypatch):
    assert config.law_search() == "bm25"  # mock mode
    monkeypatch.delenv("SC_MOCK")
    monkeypatch.setenv("GCP_PROJECT", "p")
    assert config.law_search() == "rag"
    monkeypatch.setenv("SC_GEMINI_BACKEND", "studio")
    assert config.law_search() == "bm25"
    monkeypatch.setenv("SC_LAW_SEARCH", "rag")
    assert config.law_search() == "rag"
    monkeypatch.setenv("SC_LAW_SEARCH", "x")
    with pytest.raises(ValueError):
        config.law_search()


def test_hits_map_to_provisions_and_passages(rag):
    rag.calls["contexts"] = [
        Context(rag.names["MY/OSA2025/s13"], "platform duties", 0.3),  # obligation: filtered out
        Context("t_MY_MAS1956_p011.txt", f"country: MY\n---\n{MAS_TITLE}, page 11\n\nSchedule: diabetes, cancer", 0.35),
        Context("t_MY_MAS1956_p005.txt", "country: MY\n---\ntreatment of diabetes", 0.4),
        Context(rag.names["MY/MAS1956/s3"], "advertisement ... diabetes", 0.45),  # duplicate
        Context("t_IN_DMR1954_p002.txt", "another country", 0.5),  # never for MY
        Context("unknown.txt", "not in the index", 0.5),
    ]
    hits, passages = retrieval.search_with_passages("MY", "herbal drink cures diabetes", k=3)
    assert [p.id for p in hits] == ["MY/MAS1956/s3"]  # page 5 maps to s3, no duplicates
    assert passages == [retrieval.Passage("MAS1956", MAS_TITLE, 11, "Schedule: diabetes, cancer")]
    corpus, query, top_k, location = rag.calls["args"]
    assert corpus == CORPUS and location == "asia-southeast1" and top_k >= 3

    both = retrieval.search("MY", "what must platforms do", k=5, kind=None)
    assert [p.id for p in both] == ["MY/OSA2025/s13", "MY/MAS1956/s3"]
    assert [p.id for p in retrieval.search("MY", "x", kind="obligation")] == ["MY/OSA2025/s13"]


def test_error_falls_back_to_bm25_with_note(rag):
    rag.calls["contexts"] = law_search.SearchError("HTTP 403 denied", code=403)
    notes = retrieval.track_fallbacks()
    hits = retrieval.search("MY", "cures diabetes")
    assert [p.id for p in hits] == ["MY/MAS1956/s3"]  # BM25, content only
    retrieval.search("MY", "diabetes")
    assert len(notes) == 1 and "BM25" in notes[0]


def test_no_corpus_or_bm25_mode_skips_service(rag, monkeypatch):
    assert retrieval.search("IN", "anything") == []
    assert rag.calls["n"] == 0  # IN has no corpus: BM25
    monkeypatch.setenv("SC_LAW_SEARCH", "bm25")
    assert [p.id for p in retrieval.search("MY", "diabetes")] == ["MY/MAS1956/s3"]
    assert rag.calls["n"] == 0


def test_search_law_tool_returns_passages(rag):
    rag.calls["contexts"] = [Context("t_MY_MAS1956_p011.txt", "Schedule: diabetes", 0.3)]
    _, search_law = make_tools("MY")
    out = asyncio.run(search_law("MY", "diabetes"))
    assert out == {"provisions": [], "passages": [
        {"law": MAS_TITLE, "page": 11, "snippet": "Schedule: diabetes"}]}
    assert asyncio.run(search_law("IN", "x"))["error"]


def test_retrieve_parses_response(monkeypatch):
    seen = {}

    class FakeSession:
        def post(self, url, json, timeout):
            seen["url"], seen["body"] = url, json
            return SimpleNamespace(status_code=200, content=b"x", json=lambda: {"contexts": {"contexts": [
                {"sourceUri": "gs://b/rag-staging/x/my/p_MY_A_s1_ab.txt", "text": "t", "score": 0.2}]}})

    monkeypatch.setattr(law_search, "_session", FakeSession())
    out = law_search.retrieve(CORPUS, "q", 4, location="asia-southeast1")
    assert out == [Context("p_MY_A_s1_ab.txt", "t", 0.2)]
    assert seen["url"] == ("https://asia-southeast1-aiplatform.googleapis.com/v1/"
                           "projects/1/locations/asia-southeast1:retrieveContexts")
    assert seen["body"]["vertex_rag_store"]["rag_resources"] == [{"rag_corpus": CORPUS}]
    assert seen["body"]["query"]["rag_retrieval_config"] == {
        "top_k": 4, "filter": {"vector_distance_threshold": 0.6}}


def test_index_docs_and_plan(real_provisions):
    docs = index_laws.build_docs(["MY", "IN"])
    assert set(docs) == {"MY", "IN"}
    n_prov = {cc: len(provisions.provisions_for(cc)) for cc in docs}
    for cc, d in docs.items():
        provs = [m for _, m in d.values() if m["doc_type"] == "provision"]
        assert len(provs) == n_prov[cc]  # unique names, one per provision
        assert all(m["country"] == cc for _, m in d.values())
        assert {m["kind"] for m in provs} <= {"content", "obligation"}
        assert any(m["doc_type"] == "full_text" for _, m in d.values())
    text, meta = next(v for v in docs["IN"].values() if v[1].get("provision_id") == "IN/PROGA2025/s6")
    assert "provision_id: IN/PROGA2025/s6" in text and meta["law_id"] == "PROGA2025"

    desired = {"a.txt": ("A", {}), "b.txt": ("B2", {}), "c.txt": ("C", {})}
    recorded = {"a.txt": {"hash": index_laws.content_hash("A")}, "b.txt": {"hash": index_laws.content_hash("B")},
                "c.txt": {"hash": index_laws.content_hash("C")}, "old.txt": {"hash": "x"}}
    remote = {"a.txt": "rf/a", "b.txt": "rf/b", "old.txt": "rf/old"}
    upload, delete = index_laws.plan(desired, recorded, remote)
    assert upload == ["b.txt", "c.txt"]  # changed, and missing remotely
    assert delete == ["rf/b", "rf/old"]  # old copy of the changed one, and the removed one
    assert index_laws.plan(desired, recorded | {"b.txt": {"hash": index_laws.content_hash("B2")}},
                           remote | {"c.txt": "rf/c"}) == ([], ["rf/old"])


def test_indexer_skips_unverified_laws_and_removes_their_docs(real_provisions):
    docs = index_laws.build_docs(["IN", "PK"])
    assert set(docs) == {"IN", "PK"} and docs["PK"] == {}  # PK is synced (emptied), not skipped
    assert not any(m.get("law_id") == "DMR1954" for _, m in docs["IN"].values())
    assert any(m.get("law_id") == "PROGA2025" for _, m in docs["IN"].values())
    recorded = {"t_IN_DMR1954_p002.txt": {"country": "IN", "hash": "x"}}
    remote = {"t_IN_DMR1954_p002.txt": "ragFiles/1"}
    up, delete = index_laws.plan(docs["IN"], recorded, remote)
    assert "ragFiles/1" in delete and "t_IN_DMR1954_p002.txt" not in up
