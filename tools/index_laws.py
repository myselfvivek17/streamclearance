"""Sync the law library into the Google document search (Vertex AI RAG Engine).

One-shot and idempotent. Per country (one corpus each, `<SC_SEARCH_STORE>-<cc>`,
created if missing) it indexes:
- one document per provision (law title, section, summary, verbatim text,
  exceptions, taxonomy labels), metadata country, law_id, provision_id,
  law_status, kind (content|obligation), review;
- the full text of every law in the registry, one document per page
  (data/laws/text/*.json), metadata country, law_id, doc_type=full_text, page.

Only laws with `source_verified: true` in the registry are indexed. Any other law
is logged and skipped, and its documents are deleted from the corpus if an
earlier run indexed them (a country left with nothing is emptied, not skipped).

What was uploaded (file name -> content hash + metadata) is recorded in
data/search_index.json (committed, no secrets), so a re-run only imports new or
changed documents and deletes removed ones. The file names are the keys the app
uses to map a hit back to its provision or law page (app/retrieval.py).

Needs GOOGLE_APPLICATION_CREDENTIALS (or gcloud ADC) and GOOGLE_CLOUD_PROJECT.

Usage: python -m tools.index_laws [--country MY ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time

from app import config, law_search, provisions, taxonomy
from app.schemas import Provision

log = logging.getLogger("index_laws")


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", s).strip("_")


def provision_doc_name(p: Provision) -> str:
    # ids like "IN/PROGA2025/s9(2)" are not file-name safe; the hash keeps names unique
    h = hashlib.sha1(p.id.encode()).hexdigest()[:8]
    return f"p_{p.country}_{_safe(p.law_id)}_{_safe(p.section)[:40]}_{h}.txt"


def page_doc_name(country: str, law_id: str, page: int) -> str:
    return f"t_{country}_{_safe(law_id)}_p{page:03d}.txt"


def _header(meta: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n---\n"


def provision_doc(p: Provision) -> tuple[str, dict]:
    meta = {"country": p.country, "law_id": p.law_id, "provision_id": p.id,
            "law_status": p.law_status, "kind": p.kind, "review": p.review.status,
            "page": p.page, "doc_type": "provision"}
    labels = "; ".join(f"{taxonomy.label(t)} ({t})" for t in p.targets)
    body = [f"{p.law_title}, {p.section} (page {p.page})",
            f"Summary: {p.summary}" if p.summary else "",
            f"Text: {p.verbatim_text}",
            ("Exceptions: " + "; ".join(p.exceptions)) if p.exceptions else "",
            f"Covers: {labels}" if labels else "",
            f"Duty holder: {p.duty_holder}"]
    return _header(meta) + "\n".join(b for b in body if b) + "\n", meta


def page_docs(law: provisions.Law) -> list[tuple[str, str, dict]]:
    if not law.text_path.exists():
        log.warning("%s: no text file %s", law.key, law.text_path.name)
        return []
    out = []
    for page in json.loads(law.text_path.read_text(encoding="utf-8")):
        text = page["text"].strip()
        if not text:
            continue
        meta = {"country": law.country, "law_id": law.id, "law_status": law.status,
                "doc_type": "full_text", "page": page["page"]}
        doc = _header(meta) + f"{law.title}, page {page['page']}\n\n{text}\n"
        out.append((page_doc_name(law.country, law.id, page["page"]), doc, meta))
    return out


def build_docs(countries: list[str] | None = None) -> dict[str, dict[str, tuple[str, dict]]]:
    """{country: {file name: (text, metadata)}} for every law in the registry."""
    out: dict[str, dict[str, tuple[str, dict]]] = {}
    for law in provisions.registry():
        if countries and law.country not in countries:
            continue
        docs = out.setdefault(law.country, {})  # even when empty: removes what was indexed
        if not law.source_verified:
            log.warning("%s: source not verified, not indexed (removed if it was)", law.key)
            continue
        for name, text, meta in page_docs(law):
            docs[name] = (text, meta)
    for cc, provs in provisions.all_provisions().items():
        if countries and cc not in countries:
            continue
        for p in provs:
            text, meta = provision_doc(p)
            out.setdefault(cc, {})[provision_doc_name(p)] = (text, meta)
    return out


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def plan(desired: dict[str, tuple[str, dict]], recorded: dict[str, dict],
         remote: dict[str, str]) -> tuple[list[str], list[str]]:
    """(file names to import, rag file names to delete) for one country.

    Import: new, changed (hash differs from the record) or missing remotely.
    Delete: remote files no longer wanted, and the old copy of changed ones.
    """
    upload = [n for n, (text, _) in desired.items()
              if n not in remote or recorded.get(n, {}).get("hash") != content_hash(text)]
    delete = [rf for n, rf in remote.items() if n not in desired or n in upload]
    return sorted(upload), sorted(delete)


def dump_index(index: dict) -> str:
    """JSON with one line per document, so diffs show exactly what changed."""
    head = {k: v for k, v in index.items() if k != "docs"}
    lines = [f" {json.dumps(n)}: {json.dumps(m, ensure_ascii=False, sort_keys=True)}"
             for n, m in sorted(index["docs"].items())]
    return (json.dumps(head, indent=1, ensure_ascii=False)[:-2] + ',\n "docs": {\n'
            + ",\n".join(lines) + "\n }\n}\n")


def sync(countries: list[str] | None = None, *, dry_run: bool = False) -> dict:
    project = config.gcp_project()
    if not project and not dry_run:
        raise SystemExit("set GOOGLE_CLOUD_PROJECT")
    location = config.search_location()
    old = law_search.load_index(config.SEARCH_INDEX_PATH)
    index = {"service": "vertex-ai-rag-engine", "location": location,
             "store": config.SEARCH_STORE, "corpora": dict(old.get("corpora", {})),
             "docs": dict(old.get("docs", {}))}
    if old.get("location") not in (None, location) or old.get("store") not in (None, config.SEARCH_STORE):
        index["corpora"], index["docs"] = {}, {}  # another store: start over
    summary = {}
    for cc, desired in sorted(build_docs(countries).items()):
        recorded = {n: m for n, m in index["docs"].items() if m.get("country") == cc}
        if dry_run:
            up, _ = plan(desired, recorded, {n: n for n in recorded})
            summary[cc] = {"docs": len(desired), "to_import": len(up),
                           "to_delete": len(set(recorded) - set(desired))}
            continue
        corpus = law_search.ensure_corpus(f"{config.SEARCH_STORE}-{cc.lower()}",
                                          project=project, location=location)
        index["corpora"][cc] = corpus
        remote = law_search.list_files(corpus, location=location)
        up, delete = plan(desired, recorded, remote)
        log.info("%s: %d docs, %d remote, import %d, delete %d", cc, len(desired), len(remote),
                 len(up), len(delete))
        for rf in delete:
            law_search.delete_file(rf, location=location)
        failed: list[str] = []
        wanted = len(up)
        if up:
            prefix = f"rag-staging/{time.strftime('%Y%m%d-%H%M%S')}/{cc.lower()}"
            for attempt in range(2):
                resp = law_search.import_texts(corpus, {n: desired[n][0] for n in up},
                                               location=location, bucket=config.upload_bucket(),
                                               prefix=f"{prefix}-{attempt}")
                log.info("%s: import response %s", cc, resp)
                remote = law_search.list_files(corpus, location=location)
                up = [n for n in up if n not in remote]
                if not up:
                    break
            failed = up
        remote = law_search.list_files(corpus, location=location)
        for n in [n for n in index["docs"] if index["docs"][n].get("country") == cc]:
            if n not in desired or n not in remote:
                del index["docs"][n]
        for n, (text, meta) in desired.items():
            if n in remote:
                index["docs"][n] = {"hash": content_hash(text), **meta}
        summary[cc] = {"docs": len(desired), "indexed": sum(1 for n in desired if n in remote),
                       "imported": wanted - len(failed),
                       "deleted": len(delete), "failed": failed}
    if not dry_run:
        config.SEARCH_INDEX_PATH.write_text(dump_index(index), encoding="utf-8")
        law_search.reset()
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--country", action="append", help="only this country (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="count documents, touch nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    summary = sync([c.upper() for c in args.country] if args.country else None, dry_run=args.dry_run)
    print(json.dumps(summary, indent=1))
    return 1 if any(s.get("failed") for s in summary.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
