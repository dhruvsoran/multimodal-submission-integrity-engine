"""DOCX extractor: paragraph-level extraction with estimated pagination."""

from __future__ import annotations

import os
from pathlib import Path

from ..models import ExtractedDoc
from ..normalize import (
    chunk_units,
    clean_text,
    docx_page_estimate,
    split_sentences,
)
from ..utils import get_logger

logger = get_logger(__name__)


def extract_docx(
    path: str | Path,
    chunk_target_words: int = 120,
    chunk_overlap_words: int = 35,
    min_chunk_words: int = 12,
) -> ExtractedDoc:
    """Extract chunkable text from a .docx submission.

    DOCX has no fixed pagination, so page numbers are estimated by grouping
    paragraphs (see :func:`normalize.docx_page_estimate`).  Malformed files
    return ``ok=False`` instead of raising.
    """
    path = Path(path)
    doc = ExtractedDoc(file_name=path.name, file_path=str(path), modality="docx")
    try:
        if path.stat().st_size == 0:
            doc.ok = False
            doc.error = "File is empty (zero bytes) - corrupted submission."
            logger.warning(f"[docx] {path.name}: {doc.error}")
            return doc
        import docx  # python-docx

        document = docx.Document(str(path))
        paras = [p.text for p in document.paragraphs]
        paras = [p for p in paras if p and p.strip()]
        if not paras:
            doc.error = "Document contains no readable text."
            logger.warning(f"[docx] {path.name}: {doc.error}")
            return doc
        # Tag every sentence with its estimated page.
        units = []
        for idx, para in enumerate(paras):
            page = docx_page_estimate(idx)
            for sent in split_sentences(clean_text(para)):
                units.append((sent, page, None, None))
        joined = "\n".join(clean_text(p) for p in paras)
        doc.text_length = len(joined)
        doc.full_text = joined
        doc.n_pages = docx_page_estimate(len(paras))
        doc.chunks = chunk_units(
            units,
            path.name,
            "docx",
            target_words=chunk_target_words,
            overlap_words=chunk_overlap_words,
            min_words=min_chunk_words,
        )
        logger.info(f"[docx] {path.name}: ~{doc.n_pages} pages -> {len(doc.chunks)} chunks")
        return doc
    except Exception as exc:  # noqa: BLE001
        doc.error = f"Failed to parse DOCX: {exc}"
        logger.warning(f"[docx] {path.name}: {doc.error}")
        return doc


def extract_docx_bytes(name: str, data: bytes) -> ExtractedDoc:
    """Extract from in-memory bytes."""
    import io
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as fh:
        fh.write(data)
        tmp = fh.name
    try:
        doc = extract_docx(tmp)
        doc.file_name = name
        doc.file_path = name
        return doc
    finally:
        os.unlink(tmp)
