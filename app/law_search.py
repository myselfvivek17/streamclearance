"""Vertex AI RAG Engine client: the Google document search behind retrieval path B.

Why RAG Engine and not Gemini File Search: File Search (`file_search_stores`,
the `FileSearch` tool) is Gemini Developer API only. google-genai 2.28 raises
"only supported in Gemini Developer API mode" for every store method and for
the tool on a `vertexai=True` client, and Vertex has no fileSearchStores
endpoint (HTTP 404). This app runs on Vertex AI, so it uses RAG Engine.

Layout: one corpus per country, display name `<SC_SEARCH_STORE>-<cc>` in
SC_SEARCH_LOCATION (default asia-southeast1). The corpus is the country
filter, so a query can never return another country's law. Each document's
metadata (law_id, provision_id, page, ...) is written as a header in the
document and recorded in data/search_index.json, keyed by file name; a hit's
`sourceUri` file name maps it back. Documents are imported in bulk from a
staging prefix in the GCS upload bucket (direct upload is ~30 s per file).

Plain REST with google-auth (no extra dependency). Credentials come from ADC:
GOOGLE_APPLICATION_CREDENTIALS locally, the attached service account on Cloud Run.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import config

log = logging.getLogger(__name__)

QUERY_TIMEOUT_S = 20
OP_TIMEOUT_S = 1800


class SearchError(RuntimeError):
    def __init__(self, msg: str, code: int | None = None):
        super().__init__(msg)
        self.code = code


@dataclass(frozen=True)
class Context:
    doc: str  # file name, the key in data/search_index.json "docs"
    text: str
    score: float | None  # vector distance: lower is closer


_session = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    with _session_lock:
        if _session is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            _session = AuthorizedSession(creds)
        return _session


def _host(location: str) -> str:
    return f"https://{location}-aiplatform.googleapis.com"


def _check(r, what: str) -> dict:
    if r.status_code >= 300:
        try:
            msg = r.json().get("error", {}).get("message", "")
        except ValueError:
            msg = ""
        raise SearchError(f"{what}: HTTP {r.status_code} {msg[:300]}", code=r.status_code)
    return r.json() if r.content else {}


# ---------------------------------------------------------------- query (runtime)

def retrieve(corpus: str, query: str, top_k: int, *, location: str | None = None) -> list[Context]:
    """Nearest chunks of one corpus for `query`, closest first."""
    from app import gemini

    location = location or config.search_location()
    parent = corpus.split("/ragCorpora/")[0]
    body = {"vertex_rag_store": {"rag_resources": [{"rag_corpus": corpus}]},
            "query": {"text": query[:2000], "rag_retrieval_config": {"top_k": top_k}}}

    def call() -> dict:
        r = _get_session().post(f"{_host(location)}/v1/{parent}:retrieveContexts", json=body,
                                timeout=QUERY_TIMEOUT_S)
        return _check(r, "retrieveContexts")

    data = gemini.with_retry(call, what="rag.retrieveContexts", retries=2)
    out = []
    for c in data.get("contexts", {}).get("contexts", []):
        uri = c.get("sourceUri") or c.get("sourceDisplayName") or ""
        out.append(Context(doc=uri.rsplit("/", 1)[-1], text=c.get("text", ""), score=c.get("score")))
    return out


# ---------------------------------------------------------------- admin (tools/index_laws.py)

def _wait(op: dict, location: str, what: str) -> dict:
    deadline = time.monotonic() + OP_TIMEOUT_S
    s = _get_session()
    while not op.get("done"):
        if time.monotonic() > deadline:
            raise SearchError(f"{what}: operation timed out")
        time.sleep(5)
        op = _check(s.get(f"{_host(location)}/v1/{op['name']}", timeout=60), what)
    if "error" in op:
        raise SearchError(f"{what}: {op['error']}")
    return op.get("response", {})


def _list(url: str, key: str) -> list[dict]:
    s, out, token = _get_session(), [], None
    while True:
        params = {"pageSize": 100, **({"pageToken": token} if token else {})}
        data = _check(s.get(url, params=params, timeout=60), f"list {key}")
        out += data.get(key, [])
        token = data.get("nextPageToken")
        if not token:
            return out


def ensure_corpus(display_name: str, *, project: str, location: str) -> str:
    """Resource name of the corpus with this display name, created if missing."""
    base = f"{_host(location)}/v1/projects/{project}/locations/{location}"
    for c in _list(f"{base}/ragCorpora", "ragCorpora"):
        if c.get("displayName") == display_name:
            return c["name"]
    log.info("creating RAG corpus %s in %s", display_name, location)
    op = _check(_get_session().post(f"{base}/ragCorpora", json={"displayName": display_name},
                                    timeout=60), "create corpus")
    return _wait(op, location, "create corpus")["name"]


def list_files(corpus: str, *, location: str) -> dict[str, str]:
    """{display name: rag file resource name}."""
    return {f["displayName"]: f["name"]
            for f in _list(f"{_host(location)}/v1/{corpus}/ragFiles", "ragFiles")}


def delete_file(name: str, *, location: str) -> None:
    op = _check(_get_session().delete(f"{_host(location)}/v1/{name}", timeout=60), "delete file")
    if op.get("name") and not op.get("done"):
        _wait(op, location, "delete file")


GCS_UPLOAD_URL = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
GCS_OBJECT_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o/{name}"


def import_texts(corpus: str, docs: dict[str, str], *, location: str, bucket: str,
                 prefix: str) -> dict:
    """Upload {file name: text} to gs://bucket/prefix/ and import them into the corpus.

    The staging objects are deleted afterwards (the bucket also expires them in a day).
    Returns the import response (importedRagFilesCount, failedRagFilesCount, ...).
    """
    from urllib.parse import quote

    s = _get_session()
    names = []
    try:
        for fname, text in docs.items():
            name = f"{prefix.rstrip('/')}/{fname}"
            r = s.post(GCS_UPLOAD_URL.format(bucket=bucket),
                       params={"uploadType": "media", "name": name},
                       headers={"Content-Type": "text/plain; charset=utf-8"},
                       data=text.encode("utf-8"), timeout=120)
            _check(r, f"gcs upload {fname}")
            names.append(name)
        body = {"import_rag_files_config": {
            "gcs_source": {"uris": [f"gs://{bucket}/{prefix.rstrip('/')}/"]},
            "rag_file_transformation_config": {"rag_file_chunking_config": {
                "fixed_length_chunking": {"chunk_size": 1024, "chunk_overlap": 128}}},
        }}
        op = _check(s.post(f"{_host(location)}/v1/{corpus}/ragFiles:import", json=body, timeout=120),
                    "import files")
        return _wait(op, location, "import files")
    finally:
        for name in names:
            try:
                s.delete(GCS_OBJECT_URL.format(bucket=bucket, name=quote(name, safe="")), timeout=60)
            except Exception:  # noqa: BLE001
                log.warning("could not delete gs://%s/%s", bucket, name)


# ---------------------------------------------------------------- the committed index

_index_lock = threading.Lock()
_index: dict[str, Any] | None = None


def load_index(path: Path | None = None) -> dict[str, Any]:
    """data/search_index.json: {"corpora": {cc: corpus}, "docs": {file name: metadata}}."""
    global _index
    with _index_lock:
        if path is not None:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if _index is None:
            p = config.SEARCH_INDEX_PATH
            _index = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return _index


def reset() -> None:
    global _index
    with _index_lock:
        _index = None
