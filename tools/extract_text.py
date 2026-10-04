"""Extract per-page text from each law PDF in data/laws/raw/.

Writes data/laws/text/<name>.json as a list of {"page": int, "text": str}
(1-based page numbers, matching the PDF viewer).

Scanned pages (no text layer) are OCR'd when `rapidocr-onnxruntime` and
poppler's `pdftoppm` are available (build time only, not an app dependency);
those pages get "ocr": true. OCR text often lacks spaces between words; quote
checks ignore whitespace (tools.build_rules.normalise), so that is harmless.

Usage: python -m tools.extract_text [name ...]   (names = PDF stems; default all)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "laws" / "raw"
TEXT_DIR = ROOT / "data" / "laws" / "text"


def ocr_page(pdf_path: Path, page: int, dpi: int = 200) -> str | None:
    """OCR one page (1-based); None when no OCR engine is available.

    Two detector passes (default settings, then tighter boxes on a larger image)
    each skip some lines of dense statute pages; lines only the second pass found
    are appended after the first pass's text, so nothing on the page is lost.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    exe = shutil.which("pdftoppm")
    if not exe:
        return None
    with tempfile.TemporaryDirectory(prefix="sc-ocr-") as tmp:
        out = Path(tmp) / "p"
        subprocess.run([exe, "-r", str(dpi), "-png", "-f", str(page), "-l", str(page),
                        "-singlefile", str(pdf_path), str(out)], check=True)
        passes = [[text for _box, text, _conf in (engine(str(out.with_suffix(".png")))[0] or [])]
                  for engine in _ocr_engines(RapidOCR)]
    lines = passes[0]
    seen = {_key(x) for x in lines}
    extra = [x for x in passes[1] if _key(x) not in seen and len(_key(x)) > 3]
    return "\n".join(lines + extra)


def _key(line: str) -> str:
    return "".join(line.split()).lower()


_ENGINES: list = []


def _ocr_engines(cls) -> list:
    if not _ENGINES:
        _ENGINES.extend([cls(), cls(det_limit_side_len=1280, det_unclip_ratio=1.3)])
    return _ENGINES


def extract_pdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        entry = {"page": i, "text": text}
        if not text.strip():
            ocr = ocr_page(pdf_path, i)
            if ocr:
                entry = {"page": i, "text": ocr, "ocr": True}
        pages.append(entry)
    return pages


def main(argv: list[str] | None = None, raw_dir: Path = RAW_DIR, text_dir: Path = TEXT_DIR) -> int:
    names = set(sys.argv[1:] if argv is None else argv)
    text_dir.mkdir(parents=True, exist_ok=True)
    pdfs = [p for p in sorted(raw_dir.glob("*.pdf")) if not names or p.stem in names]
    if not pdfs:
        print(f"No PDFs found in {raw_dir}", file=sys.stderr)
        return 1
    for pdf in pdfs:
        pages = extract_pdf(pdf)
        out = text_dir / f"{pdf.stem}.json"
        out.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
        empty = [p["page"] for p in pages if not p["text"].strip()]
        ocr = [p["page"] for p in pages if p.get("ocr")]
        chars = sum(len(p["text"]) for p in pages)
        print(f"{pdf.stem}: {len(pages)} pages, {chars} chars, empty pages: {empty or 'none'}"
              + (f", OCR pages: {ocr}" if ocr else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
