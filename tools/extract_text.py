"""Extract per-page text from each law PDF in data/laws/raw/.

Writes data/laws/text/<name>.json as a list of {"page": int, "text": str}
(1-based page numbers, matching the PDF viewer).

Usage: python -m tools.extract_text
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "laws" / "raw"
TEXT_DIR = ROOT / "data" / "laws" / "text"


def extract_pdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    return [
        {"page": i, "text": page.extract_text() or ""}
        for i, page in enumerate(reader.pages, start=1)
    ]


def main(raw_dir: Path = RAW_DIR, text_dir: Path = TEXT_DIR) -> int:
    text_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {raw_dir}", file=sys.stderr)
        return 1
    for pdf in pdfs:
        pages = extract_pdf(pdf)
        out = text_dir / f"{pdf.stem}.json"
        out.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
        empty = [p["page"] for p in pages if not p["text"].strip()]
        chars = sum(len(p["text"]) for p in pages)
        print(f"{pdf.stem}: {len(pages)} pages, {chars} chars, empty pages: {empty or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
