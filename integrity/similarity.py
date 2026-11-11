"""Peer-to-peer similarity, boilerplate detection and citation alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .models import Chunk, fmt_range
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class Alignment:
    """One chunk-level match between a query chunk and a source chunk."""

    query_chunk: Chunk
    matched_chunk: Chunk
    similarity: float
    lexical_jaccard: float


@dataclass
class PairMatch:
    """Document-pair level result."""

    query_file: str
    matched_file: str
    coverage: float  # fraction of query words aligned at >= threshold
    alignments: List[Alignment] = field(default_factory=list)
    cross_modal: bool = False

    def top_alignments(self, n: int = 3) -> List[Alignment]:
        return sorted(self.alignments, key=lambda a: a.similarity, reverse=True)[:n]


def _lexical_jaccard(a: str, b: str) -> float:
    wa = set(w.lower() for w in a.split() if len(w) > 2)
    wb = set(w.lower() for w in b.split() if len(w) > 2)
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


def best_matches_for_chunks(
    query_chunks: Sequence[Chunk],
    query_vecs: np.ndarray,
    corpus_vecs: np.ndarray,
    chunk_ids: Sequence[str],
    chunk_lookup: Dict[str, Chunk],
    top_k: int = 5,
    exclude_self_file: bool = True,
) -> List[List[Alignment]]:
    """For every query chunk, find its best matching corpus chunk per file.

    Returns, for each query chunk, the best alignment against *every other
    file* (not just a global top-k) so document-pair coverage is exact.
    """
    n_corp = len(chunk_ids)
    if n_corp == 0 or len(query_chunks) == 0:
        return [[] for _ in query_chunks]
    sims = query_vecs @ corpus_vecs.T  # (n_query, n_corpus) exact cosine
    out: List[List[Alignment]] = []
    for qi in range(len(query_chunks)):
        q = query_chunks[qi]
        row = sims[qi]
        seen: Dict[str, Alignment] = {}
        order = np.argsort(-row)
        for ci in order[: min(len(row), top_k * 8)]:
            cand = chunk_lookup[chunk_ids[int(ci)]]
            if exclude_self_file and cand.source_file == q.source_file:
                continue  # self-match exclusion (assignment requirement)
            if cand.source_file not in seen:
                seen[cand.source_file] = Alignment(
                    query_chunk=q,
                    matched_chunk=cand,
                    similarity=float(row[ci]),
                    lexical_jaccard=_lexical_jaccard(q.text, cand.text),
                )
        out.append(list(seen.values()))
    return out


def find_repeated_sentences(
    docs: Sequence["ExtractedDoc"],  # noqa: F821
    min_files: int = 3,
    min_sentence_words: int = 6,
) -> Dict[str, int]:
    """Count, across the corpus, how many files contain each normalized sentence.

    Returns {sentence_hash: n_files}.  Sentences repeated in >= ``min_files``
    distinct files are institutional boilerplate (case prompts, headers) -
    Scenario 4 protection.  Exact-match based, so genuinely copied *content*
    between exactly 2 files is never caught; only corpus-wide template text.
    """
    import re as _re

    def norm_sent(s: str) -> str:
        return _re.sub(r"\W+", " ", s.lower()).strip()

    counts: Dict[str, set] = {}
    for d in docs:
        for chunk in d.chunks:
            for sent in _re.split(r"(?<=[.!?])\s+", chunk.text):
                words = norm_sent(sent).split()
                if len(words) < min_sentence_words:
                    continue
                key = " ".join(words)
                counts.setdefault(key, set()).add(d.file_name)
    return {k: len(v) for k, v in counts.items() if len(v) >= min_files}


def mark_repeated_template_chunks(
    docs: Sequence["ExtractedDoc"],
    repeated: Dict[str, int],
    fraction_threshold: float = 0.5,
) -> Dict[str, int]:
    """Mark chunks that are mostly composed of corpus-wide template sentences.

    A chunk whose repeated-template words exceed ``fraction_threshold`` of its
    words is discounted entirely.  Returns summary stats.
    """
    import re as _re

    def norm_sent(s: str) -> str:
        return _re.sub(r"\W+", " ", s.lower()).strip()

    marked_chunks, marked_words = 0, 0
    for d in docs:
        for chunk in d.chunks:
            sents = [s for s in _re.split(r"(?<=[.!?])\s+", chunk.text) if s.strip()]
            template_words = 0
            for sent in sents:
                key = " ".join(norm_sent(sent).split())
                if key in repeated:
                    template_words += len(key.split())
            frac = template_words / max(1, chunk.n_words)
            if frac >= fraction_threshold and template_words >= 20:
                if not chunk.is_boilerplate:
                    marked_chunks += 1
                    marked_words += chunk.n_words
                chunk.is_boilerplate = True
                chunk.boilerplate_group = chunk.boilerplate_group or "repeated_template"
    return {"extra_chunks": marked_chunks, "extra_words": marked_words}


def find_boilerplate(
    all_chunks: Sequence[Chunk],
    vecs: np.ndarray,
    similarity_threshold: float = 0.90,
    min_files: int = 3,
) -> Dict[str, str]:
    """Mark chunks that near-duplicate across many files as boilerplate.

    Greedy clustering: any cluster of chunks with pairwise cosine >=
    ``similarity_threshold`` that spans >= ``min_files`` distinct files is
    treated as shared institutional template (case prompts, headers).
    Boilerplate is discounted from peer-similarity scoring and citations
    (Scenario 4: benign common formatting).
    Returns {chunk_id: group_label}.
    """
    marks: Dict[str, str] = {}
    n = len(all_chunks)
    if n == 0:
        return marks
    sims = vecs @ vecs.T
    assigned = np.array([c.is_boilerplate for c in all_chunks], dtype=bool)
    group = 0
    for i in range(n):
        if assigned[i]:
            continue
        members = np.where(sims[i] >= similarity_threshold)[0]
        members = [j for j in members if not assigned[j]]
        files = {all_chunks[j].source_file for j in members}
        if len(files) >= min_files:
            group += 1
            label = f"template_{group:02d}"
            for j in members:
                assigned[j] = True
                marks[all_chunks[j].chunk_id] = label
                all_chunks[j].is_boilerplate = True
                all_chunks[j].boilerplate_group = label
            logger.info(
                f"[boilerplate] {label}: {len(members)} chunks across "
                f"{len(files)} files discounted (e.g. '{all_chunks[members[0]].text[:60]}...')"
            )
    return marks


def pairwise_coverage(
    alignments_per_chunk: List[List[Alignment]],
    query_chunks: Sequence[Chunk],
    threshold: float,
) -> Dict[str, PairMatch]:
    """Aggregate chunk alignments into document-pair coverage scores.

    coverage = (words in query chunks whose best match to the other document
    clears ``threshold``) / (total non-boilerplate words in the document).
    Self-matches are excluded upstream, identical-file matches never occur.
    """
    totals: Dict[str, Dict[str, float]] = {}
    aligned_words: Dict[str, Dict[str, float]] = {}
    for chunk, aligns in zip(query_chunks, alignments_per_chunk):
        if chunk.is_boilerplate:
            continue
        weight = float(chunk.n_words)
        for a in aligns:
            if a.matched_chunk.is_boilerplate:
                continue
            totals.setdefault(a.matched_chunk.source_file, {})
            aligned_words.setdefault(a.matched_chunk.source_file, {})
            totals[a.matched_chunk.source_file][chunk.chunk_id] = weight
            if a.similarity >= threshold:
                aligned_words[a.matched_chunk.source_file][chunk.chunk_id] = weight
    results: Dict[str, PairMatch] = {}
    for file, per_chunk in totals.items():
        denom = sum(per_chunk.values())
        num = sum(aligned_words.get(file, {}).values())
        coverage = 100.0 * num / denom if denom else 0.0
        aligns = [
            x
            for chunk_aligns in alignments_per_chunk
            for x in chunk_aligns
            if x.matched_chunk.source_file == file
            and x.similarity >= threshold
            and not x.query_chunk.is_boilerplate
            and not x.matched_chunk.is_boilerplate
        ]
        results[file] = PairMatch(
            query_file=query_chunks[0].source_file if query_chunks else "",
            matched_file=file,
            coverage=round(coverage, 1),
            alignments=aligns,
            cross_modal=any(
                a.query_chunk.modality != a.matched_chunk.modality for a in aligns
            ),
        )
    return results


def alignment_to_dict(a: Alignment) -> dict:
    """Serialize one alignment following the assignment JSON schema."""
    q, m = a.query_chunk, a.matched_chunk
    loc: Dict[str, object] = {}
    if q.modality == "mp4":
        loc["query_timestamp"] = fmt_range(q.start_ts, q.end_ts)
    else:
        loc["query_page"] = q.page
    if m.modality == "mp4":
        loc["matched_timestamp"] = fmt_range(m.start_ts, m.end_ts)
    else:
        loc["matched_page"] = m.page
    return {
        "matched_file": m.source_file,
        "similarity_score": round(a.similarity, 3),
        "query_passage": q.text,
        "matched_passage": m.text,
        "location": loc,
        "lexical_jaccard": round(a.lexical_jaccard, 3),
        "query_chunk_id": q.chunk_id,
        "matched_chunk_id": m.chunk_id,
    }
