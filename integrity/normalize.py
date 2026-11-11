"""Text cleaning, normalization and chunking utilities.

All extractors funnel their raw text through this module so that every
modality (PDF / DOCX / MP4 transcript) is compared on equal footing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

from .models import Chunk

# A "unit" is one sentence plus its metadata:
# (sentence_text, page_no_or_None, start_ts_or_None, end_ts_or_None)
Unit = Tuple[str, Optional[int], Optional[float], Optional[float]]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])+")
_PAGE_NUM_LINE = re.compile(r"^\s*(?:page\s*)?\d{1,3}\s*(?:/|of\s*)?\s*\d{0,3}\s*$", re.I)
_HYPHEN_JOIN = re.compile(r"(\w)-\n(\w)")


def clean_text(raw: str) -> str:
    """Normalize unicode, re-flow line breaks and collapse whitespace.

    Steps:
    1. NFKC unicode folding (resolves ligatures such as "ﬁ" -> "fi").
    2. Re-join words broken by hyphenated line wraps ("sup-\\nply" -> "supply").
    3. Collapse runs of spaces; keep paragraph boundaries as single newlines.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = _HYPHEN_JOIN.sub(r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip page-number-only lines left over from PDF pagination.
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not _PAGE_NUM_LINE.match(ln)]
    text = "\n".join(lines)
    # Collapse intra-paragraph whitespace.
    paragraphs = [" ".join(p.split()) for p in text.split("\n")]
    return "\n".join(p for p in paragraphs if p)


def split_sentences(text: str) -> List[str]:
    """Regex sentence splitter (good-enough for well-formed submissions)."""
    if not text.strip():
        return []
    parts = _SENT_SPLIT.split(text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def split_sentences_with_meta(
    page_texts: Sequence[Tuple[int, str]]
) -> List[Unit]:
    """Split per-page text into sentences tagged with their source page.

    ``page_texts`` is a sequence of (page_number, cleaned_page_text).
    """
    units: List[Unit] = []
    for page_no, page_text in page_texts:
        for sent in split_sentences(page_text):
            units.append((sent, page_no, None, None))
    return units


def word_count(text: str) -> int:
    return len(text.split())


def chunk_units(
    units: Sequence[Unit],
    source_file: str,
    modality: str,
    target_words: int = 170,
    overlap_words: int = 45,
    min_words: int = 12,
) -> List[Chunk]:
    """Group sentence units into overlapping chunks.

    Chunker slides a window over sentences until ``target_words`` is reached,
    then rolls back by roughly ``overlap_words`` so neighbouring chunks share
    context.  Page / timestamp metadata comes from the first and last sentence
    inside the chunk.  Sentence units shorter than ``min_words`` words are
    dropped (isolated headings, greetings, etc.).
    """
    units = [u for u in units if word_count(u[0]) >= 1]
    # Merge tiny consecutive units (ASR fragments, list items) into sentences.
    merged: List[Unit] = []
    buf_text, buf_page, buf_start, buf_end = "", None, None, None
    for text, page, start, end in units:
        if buf_text and word_count(buf_text) + word_count(text) < 8 and not buf_text.endswith((".", "!", "?")):
            buf_text = f"{buf_text} {text}".strip()
            buf_end = end if end is not None else buf_end
        else:
            if buf_text:
                merged.append((buf_text, buf_page, buf_start, buf_end))
            buf_text, buf_page, buf_start, buf_end = text, page, start, end
    if buf_text:
        merged.append((buf_text, buf_page, buf_start, buf_end))

    # Pre-compute word counts.
    counts = [word_count(u[0]) for u in merged]
    n = len(merged)
    chunks: List[Chunk] = []
    start = 0
    while start < n:
        total, end = 0, start
        while end < n and total < target_words:
            total += counts[end]
            end += 1
        if total < min_words:  # remainder too small to be meaningful
            break
        texts = [merged[i][0] for i in range(start, end)]
        pages = [merged[i][1] for i in range(start, end) if merged[i][1] is not None]
        starts = [merged[i][2] for i in range(start, end) if merged[i][2] is not None]
        ends = [merged[i][3] for i in range(start, end) if merged[i][3] is not None]
        text = " ".join(texts)
        chunk = Chunk(
            text=text,
            source_file=source_file,
            modality=modality,
            n_words=total,
            chunk_index=len(chunks),
            page=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            start_ts=min(starts) if starts else None,
            end_ts=max(ends) if ends else None,
        )
        chunk.chunk_id = f"{source_file}#c{chunk.chunk_index:03d}"
        chunks.append(chunk)
        # Roll the window back so the next chunk overlaps by ~overlap_words.
        back, overlap_total = end, 0
        while back - 1 > start and overlap_total + counts[back - 1] <= overlap_words:
            back -= 1
            overlap_total += counts[back]
        start = back if back > start else start + 1
    return chunks


def docx_page_estimate(paragraph_index: int, paragraphs_per_page: int = 28) -> int:
    """DOCX files carry no hard pagination.

    We approximate page numbers by grouping paragraphs (default 28 per page,
    a typical Word page of body text).  Documented as an assumption in the
    README; positions remain stable and human-verifiable either way.
    """
    return paragraph_index // paragraphs_per_page + 1
