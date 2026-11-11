"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """A normalized, metadata-bearing span of text from one submission."""

    text: str
    source_file: str
    modality: str  # "pdf" | "docx" | "mp4"
    n_words: int = 0
    chunk_index: int = 0
    chunk_id: str = ""
    # Document location (PDF: real page numbers; DOCX: estimated page).
    page: Optional[int] = None
    page_end: Optional[int] = None
    # Video location (seconds). Present only for transcript chunks.
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    # Filled in by the corpus-level boilerplate detector.
    is_boilerplate: bool = False
    boilerplate_group: Optional[str] = None

    def location(self) -> str:
        """Human-readable location string used in reports."""
        if self.modality == "mp4" and self.start_ts is not None:
            return f"timestamp {fmt_range(self.start_ts, self.end_ts)}"
        if self.page is not None:
            if self.page_end is not None and self.page_end != self.page:
                return f"pages {self.page}-{self.page_end}"
            return f"page {self.page}"
        return "location unknown"


def fmt_ts(seconds: Optional[float]) -> str:
    """Format seconds as HH:MM:SS (e.g. 84 -> "00:01:24")."""
    if seconds is None:
        return "00:00:00"
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_range(start: Optional[float], end: Optional[float]) -> str:
    """Format a time range as "00:00:24 - 00:00:36" (JSON schema format)."""
    return f"{fmt_ts(start)} - {fmt_ts(end)}"


@dataclass
class ExtractedDoc:
    """Result of parsing one submission file."""

    file_name: str
    file_path: str
    modality: str  # "pdf" | "docx" | "mp4"
    ok: bool = True
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    full_text: str = ""
    text_length: int = 0
    language: Optional[str] = None
    duration_s: Optional[float] = None
    n_pages: Optional[int] = None

    @property
    def total_chunk_words(self) -> int:
        return sum(c.n_words for c in self.chunks)
