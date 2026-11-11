"""PDF extractor: page-aware text extraction with header/footer stripping."""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import List

from ..models import ExtractedDoc
from ..normalize import clean_text, chunk_units, split_sentences_with_meta
from ..utils import get_logger

logger = get_logger(__name__)

_PAGE_ONLY = re.compile(r"^[\s\W]*\d+[\s\W]*$")


def _norm_line(line: str) -> str:
    return " ".join(line.lower().split())


def _strip_headers_footers(pages: List[str]) -> List[str]:
    """Remove lines that repeat across most pages (running headers/footers).

    A line qualifies when it appears (normalized) on >= 60% of pages (with at
    least 3 pages) or on every page when the document has exactly 2 pages.
    Bare page-number lines are always removed.  This also removes repeated
    institutional templates that would otherwise inflate similarity scores.
    """
    n = len(pages)
    # Always drop bare page-number lines ("Page 3", "3", "3 of 5", "3/5").
    cleaned = [
        "\n".join(l for l in p.split("\n") if l.strip() and not _PAGE_ONLY.match(l))
        for p in pages
    ]
    if n < 2:
        return cleaned
    first_lines: Counter = Counter()
    last_lines: Counter = Counter()
    for p in cleaned:
        lines = [l for l in p.split("\n") if l.strip()]
        if lines:
            first_lines[_norm_line(lines[0])] += 1
            last_lines[_norm_line(lines[-1])] += 1
    min_count = n if n == 2 else max(2, int(0.6 * n))

    def repeated(c: Counter) -> set:
        return {line for line, cnt in c.items() if cnt >= min_count and len(line) >= 4}

    headers, footers = repeated(first_lines), repeated(last_lines)
    stripped: List[str] = []
    for p in cleaned:
        lines = [l for l in p.split("\n") if l.strip()]
        if lines and _norm_line(lines[0]) in headers:
            lines = lines[1:]
        if lines and _norm_line(lines[-1]) in footers:
            lines = lines[:-1]
        stripped.append("\n".join(lines))
    return stripped


def extract_pdf(
    path: str | Path,
    chunk_target_words: int = 120,
    chunk_overlap_words: int = 35,
    min_chunk_words: int = 12,
) -> ExtractedDoc:
    """Extract chunkable text from a PDF.

    Never raises: any malformed / unreadable file returns ``ok=False`` with an
    informative error so the pipeline can continue with remaining documents.
    """
    path = Path(path)
    doc = ExtractedDoc(
        file_name=path.name, file_path=str(path), modality="pdf"
    )
    try:
        if path.stat().st_size == 0:
            doc.ok = False
            doc.error = "File is empty (zero bytes) - corrupted submission."
            logger.warning(f"[pdf] {path.name}: {doc.error}")
            return doc
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
                doc.warnings.append("PDF was encrypted; decrypted with empty password.")
            except Exception:
                doc.error = "PDF is password protected and cannot be read."
                logger.warning(f"[pdf] {path.name}: {doc.error}")
                return doc
        raw_pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                raw_pages.append((i, page.extract_text() or ""))
            except Exception:
                doc.warnings.append(f"Text extraction failed on page {i}; page skipped.")
                raw_pages.append((i, ""))
        doc.n_pages = len(raw_pages)
        page_texts = [(i, clean_text(t)) for i, t in raw_pages]
        # Header/footer stripping needs cross-page frequency context, so it
        # runs over the whole page set at once.
        stripped = _strip_headers_footers([t for _, t in page_texts])
        page_texts = [(i, t) for (i, _), t in zip(page_texts, stripped)]
        joined = "\n".join(t for _, t in page_texts)
        doc.text_length = len(joined)
        doc.full_text = joined
        if not joined.strip():
            doc.error = (
                "No extractable text found (file may be scanned images or blank)."
            )
            logger.warning(f"[pdf] {path.name}: {doc.error}")
            return doc
        units = split_sentences_with_meta(page_texts)
        doc.chunks = chunk_units(
            units,
            path.name,
            "pdf",
            target_words=chunk_target_words,
            overlap_words=chunk_overlap_words,
            min_words=min_chunk_words,
        )
        doc.warnings = [w for w in doc.warnings]
        logger.info(f"[pdf] {path.name}: {doc.n_pages} pages -> {len(doc.chunks)} chunks")
        return doc
    except Exception as exc:  # noqa: BLE001 - intentional: never crash the pipeline
        doc.error = f"Failed to parse PDF: {exc}"
        logger.warning(f"[pdf] {path.name}: {doc.error}")
        return doc


def extract_pdf_bytes(name: str, data: bytes) -> ExtractedDoc:
    """Extract from in-memory bytes (used by tests and sample generation)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(data)
        tmp = fh.name
    try:
        doc = extract_pdf(tmp)
        doc.file_name = name
        doc.file_path = name
        return doc
    finally:
        os.unlink(tmp)
