"""External / web reference plagiarism engine.

Full-scale web scraping is rate-limited and non-reproducible, so the default
engine matches submissions against a **local reference knowledge base** of
open, Wikipedia-style articles stored in ``reference_corpus/``.  This is the
approach the brief explicitly permits ("a local reference knowledge base ...
or a free search API with fallback mocks").

An optional ``--fetch-web`` mode extends the local KB at runtime via the free
Wikipedia REST API (no key, graceful fallback to local-only when offline).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .models import Chunk, fmt_range
from .normalize import clean_text, chunk_units, split_sentences
from .utils import get_logger

logger = get_logger(__name__)

WIKIPEDIA_TOPICS = [
    "Electric vehicle charging network",
    "Microgrid",
    "Congestion pricing",
    "Precision agriculture",
    "Microfinance",
    "Water purification",
]


@dataclass
class ExternalMatch:
    source_ref: str
    similarity: float
    query_chunk: Chunk
    matched_chunk: Chunk
    lexical_jaccard: float


@dataclass
class ExternalResult:
    coverage_percentage: float
    matches: List[ExternalMatch] = field(default_factory=list)


class ReferenceMatcher:
    """Matches submission chunks against the local reference knowledge base."""

    def __init__(
        self,
        corpus_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        chunk_target_words: int = 170,
        chunk_overlap_words: int = 45,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.chunks: List[Chunk] = []
        self.chunk_target_words = chunk_target_words
        self.chunk_overlap_words = chunk_overlap_words
        self._load_corpus()

    # ------------------------------------------------------------------ #
    def _load_corpus(self) -> None:
        files = sorted(self.corpus_dir.rglob("*.txt")) if self.corpus_dir.exists() else []
        for fp in files:
            try:
                text = clean_text(fp.read_text(encoding="utf-8", errors="ignore"))
                pages = [(i, para) for i, para in enumerate(text.split("\n"), 1) if para.strip()]
                units = [
                    (sent, page, None, None)
                    for page, para in pages
                    for sent in split_sentences(para)
                ]
                self.chunks.extend(
                    chunk_units(
                        units,
                        fp.name,
                        "reference",
                        target_words=self.chunk_target_words,
                        overlap_words=self.chunk_overlap_words,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[external] failed to load reference file {fp.name}: {exc}")
        logger.info(
            f"[external] reference KB: {len(files)} files -> {len(self.chunks)} chunks"
        )

    # ------------------------------------------------------------------ #
    def embeddings_fingerprint(self) -> str:
        """Stable hash of KB contents, used to cache reference embeddings."""
        h = hashlib.md5()
        for c in self.chunks:
            h.update(c.text.encode("utf-8"))
        return h.hexdigest()

    def cached_vectors(self, cache_dir: Path) -> Optional[np.ndarray]:
        path = cache_dir / f"reference_vectors_{self.embeddings_fingerprint()[:10]}.npy"
        if path.exists():
            try:
                vecs = np.load(path)
                if len(vecs) == len(self.chunks):
                    logger.info(f"[external] loaded cached reference vectors: {path.name}")
                    return vecs
            except Exception:  # noqa: BLE001
                pass
        return None

    def save_vectors(self, cache_dir: Path, vecs: np.ndarray) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"reference_vectors_{self.embeddings_fingerprint()[:10]}.npy"
        np.save(path, vecs)

    # ------------------------------------------------------------------ #
    def score(
        self,
        query_chunks: Sequence[Chunk],
        query_vecs: np.ndarray,
        ref_vecs: np.ndarray,
        threshold: float,
    ) -> ExternalResult:
        """Coverage of the query document by the reference KB + top matches."""
        if not self.chunks or len(query_chunks) == 0:
            return ExternalResult(coverage_percentage=0.0)
        sims = query_vecs @ ref_vecs.T  # exact cosine
        aligned_words, total_words = 0.0, 0.0
        matches: List[ExternalMatch] = []
        for qi, chunk in enumerate(query_chunks):
            if chunk.is_boilerplate:
                continue
            total_words += chunk.n_words
            row = sims[qi]
            best = int(np.argmax(row))
            best_sim = float(row[best])
            ref_chunk = self.chunks[best]
            if best_sim >= threshold and not ref_chunk.is_boilerplate:
                aligned_words += chunk.n_words
                if len(matches) < 8 or best_sim >= 0.85:
                    matches.append(
                        ExternalMatch(
                            source_ref=ref_chunk.source_file,
                            similarity=best_sim,
                            query_chunk=chunk,
                            matched_chunk=ref_chunk,
                            lexical_jaccard=_jaccard(chunk.text, ref_chunk.text),
                        )
                    )
        matches.sort(key=lambda m: m.similarity, reverse=True)
        coverage = 100.0 * aligned_words / total_words if total_words else 0.0
        return ExternalResult(coverage_percentage=round(coverage, 1), matches=matches[:8])


def _jaccard(a: str, b: str) -> float:
    wa = set(w.lower() for w in a.split() if len(w) > 2)
    wb = set(w.lower() for w in b.split() if len(w) > 2)
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


def external_match_to_dict(m: ExternalMatch) -> dict:
    q, r = m.query_chunk, m.matched_chunk
    loc: Dict[str, object] = {
        "query_page": q.page,
        "source_section": r.chunk_index,
    }
    if q.start_ts is not None:
        loc["query_timestamp"] = fmt_range(q.start_ts, q.end_ts)
    return {
        "matched_source": m.source_ref,
        "similarity_score": round(m.similarity, 3),
        "query_passage": q.text,
        "matched_passage": r.text,
        "location": loc,
        "lexical_jaccard": round(m.lexical_jaccard, 3),
    }


def fetch_wikipedia_pages(
    topics: List[str], out_dir: Path, timeout: int = 8
) -> List[str]:
    """Optionally grow the KB from Wikipedia's free REST API (no API key).

    Returns the list of newly saved files; empty on network failure.
    """
    import urllib.request

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    for topic in topics:
        slug = re.sub(r"\W+", "_", topic).strip("_")
        dest = out_dir / f"wiki_{slug}.txt"
        if dest.exists():
            continue
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + topic.replace(" ", "_")
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "integrity-engine/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload.get("extract") or ""
            if len(text.split()) >= 60:
                dest.write_text(
                    f"SOURCE: Wikipedia (retrieved at runtime) - {topic}\n\n{text}",
                    encoding="utf-8",
                )
                saved.append(dest.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[external] web fetch failed for '{topic}': {exc}")
    if saved:
        logger.info(f"[external] fetched {len(saved)} reference pages from Wikipedia")
    return saved
