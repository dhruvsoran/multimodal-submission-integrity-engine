"""End-to-end orchestration: extract -> embed -> index -> score -> report."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .ai_detector import build_ai_detector
from .embeddings import EmbeddingEngine
from .external import ReferenceMatcher
from .extractors import EXTRACTORS, supported_extensions
from .models import Chunk, ExtractedDoc
from .reporter import (
    DocAnalysis,
    build_json_report,
    build_markdown_report,
    build_summary_markdown,
    compose_explanation,
    compute_risk,
    compute_scores,
    write_json,
)
from .similarity import (
    best_matches_for_chunks,
    find_boilerplate,
    find_repeated_sentences,
    mark_repeated_template_chunks,
    pairwise_coverage,
)
from .utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "whisper_model": "base",
    "device": "cpu",
    "chunk": {
        "target_words": 120,
        "overlap_words": 35,
        "min_words": 12,
        "mp4_target_words": 100,
        "mp4_overlap_words": 25,
    },
    "similarity": {"match_threshold": 0.75, "top_k_per_chunk": 5},
    "boilerplate": {"similarity_threshold": 0.86, "min_files": 3},
    "external": {"match_threshold": 0.85, "reference_dir": "reference_corpus"},
    "ai": {"backend": "heuristic"},
    "risk": {
        "weights": {"peer": 0.5, "external": 0.3, "ai": 0.2},
        "high": 60,
        "medium": 35,
        "pair_high_floor": 70,
        "ai_medium_floor": 60,
        "external_medium_floor": 50,
    },
    "report": {
        "max_peer_matches": 6,
        "max_multimodal_matches": 6,
        "max_passages_per_pair": 3,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class IntegrityPipeline:
    """Runs the full multimodal integrity check over a folder of submissions."""

    def __init__(self, config: dict | None = None, log_file: str | Path | None = None):
        self.cfg = deep_merge(DEFAULT_CONFIG, config or {})
        self.logger = get_logger("pipeline", log_file=log_file)
        self.embedding: EmbeddingEngine | None = None
        self.reference: ReferenceMatcher | None = None

    # ------------------------------------------------------------------ #
    def _collect_files(self, input_dir: Path) -> List[Path]:
        files: List[Path] = []
        for p in sorted(input_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTRACTORS:
                files.append(p)
        if not files:
            self.logger.warning(
                f"No supported files found in {input_dir} "
                f"(supported: {', '.join(supported_extensions())})"
            )
        return files

    def _extract_all(self, files: List[Path]) -> List[ExtractedDoc]:
        cfg = self.cfg
        c = cfg["chunk"]
        chunk_kw = dict(
            chunk_target_words=c["target_words"],
            chunk_overlap_words=c["overlap_words"],
            min_chunk_words=c["min_words"],
        )
        docs: List[ExtractedDoc] = []
        for fp in files:
            extractor = EXTRACTORS[fp.suffix.lower()]
            t0 = time.time()
            try:
                if fp.suffix.lower() == ".pdf":
                    doc = extractor(fp, **chunk_kw)
                elif fp.suffix.lower() in {".docx", ".doc"}:
                    doc = extractor(fp, **chunk_kw)
                else:  # media -> ASR path
                    doc = extractor(
                        fp,
                        whisper_model=cfg["whisper_model"],
                        device=cfg["device"],
                        chunk_target_words=c["mp4_target_words"],
                        chunk_overlap_words=c["mp4_overlap_words"],
                        min_words=c["min_words"],
                    )
            except Exception as exc:  # noqa: BLE001 - never crash on one file
                doc = ExtractedDoc(
                    file_name=fp.name, file_path=str(fp), modality=fp.suffix[1:],
                    ok=False, error=f"Unexpected extractor failure: {exc}",
                )
                self.logger.warning(f"[extract] {fp.name}: {doc.error}")
            docs.append(doc)
            self.logger.info(
                f"[extract] {fp.name}: ok={doc.ok} chunks={len(doc.chunks)} "
                f"({time.time() - t0:.1f}s)"
            )
        return docs

    # ------------------------------------------------------------------ #
    def run(self, input_dir: str | Path, output_dir: str | Path) -> dict:
        input_dir, output_dir = Path(input_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        cfg = self.cfg

        # 1. Extract -------------------------------------------------------
        files = self._collect_files(input_dir)
        docs = self._extract_all(files)
        self.logger.info(f"[pipeline] extraction finished in {time.time() - t_start:.1f}s")

        # 2. Reference KB (external engine) --------------------------------
        ref_dir = Path(cfg["external"]["reference_dir"])
        if not ref_dir.is_absolute():
            ref_dir = Path(__file__).resolve().parent.parent / ref_dir
        self.reference = ReferenceMatcher(
            ref_dir,
            chunk_target_words=cfg["chunk"]["target_words"],
            chunk_overlap_words=cfg["chunk"]["overlap_words"],
        )

        # 3. Embed every chunk in ONE batch (keeps TF-IDF fallback consistent
        #    and mirrors production batch-embedding).
        sub_chunks: List[Chunk] = [c for d in docs for c in d.chunks]
        all_texts = [c.text for c in sub_chunks] + [c.text for c in self.reference.chunks]
        self.embedding = EmbeddingEngine(cfg["embedding_model"])
        vectors = (
            self.embedding.embed(all_texts)
            if all_texts
            else np.zeros((0, 1), dtype="float32")
        )
        n_sub = len(sub_chunks)
        sub_vecs = vectors[:n_sub]
        ref_vecs = vectors[n_sub:]
        if self.reference.chunks and len(ref_vecs) == len(self.reference.chunks):
            cache = Path(__file__).resolve().parent.parent / ".cache"
            self.reference.save_vectors(cache, ref_vecs)

        chunk_lookup: Dict[str, Chunk] = {c.chunk_id: c for c in sub_chunks}
        chunk_ids = [c.chunk_id for c in sub_chunks]

        # 4a. Exact repeated-template detection (Scenario 4) ----------------
        # Corpus-wide repeated sentences (e.g. identical case prompts) are
        # boilerplate even when mixed with body text inside one chunk.
        repeated = find_repeated_sentences(
            [d for d in docs if d.chunks],
            min_files=cfg["boilerplate"]["min_files"],
        )
        sent_stats = mark_repeated_template_chunks(
            [d for d in docs if d.chunks], repeated
        )
        if repeated:
            self.logger.info(
                f"[boilerplate] repeated-template sentences: {len(repeated)} "
                f"(mixed-chunk discount: +{sent_stats['extra_chunks']} chunks, "
                f"+{sent_stats['extra_words']} words)"
            )

        # 4b. Embedding-based near-duplicate template clustering ------------
        bp_marks = find_boilerplate(
            sub_chunks,
            sub_vecs,
            similarity_threshold=cfg["boilerplate"]["similarity_threshold"],
            min_files=cfg["boilerplate"]["min_files"],
        )
        boilerplate_files = len(
            {sub_chunks[i].source_file for i, c in enumerate(sub_chunks) if c.is_boilerplate}
        )
        self.logger.info(
            f"[pipeline] boilerplate chunks discounted: {len(bp_marks)} "
            f"across {boilerplate_files} files"
        )

        # 5. Peer similarity (each doc vs every other doc) ------------------
        ai_score_fn, ai_backend = build_ai_detector(cfg["ai"]["backend"])
        threshold = cfg["similarity"]["match_threshold"]
        align_map = best_matches_for_chunks(
            sub_chunks,
            sub_vecs,
            sub_vecs,
            chunk_ids,
            chunk_lookup,
            top_k=cfg["similarity"]["top_k_per_chunk"],
        )

        # Re-align per document (alignment rows follow sub_chunks order).
        analyses: List[DocAnalysis] = []
        matrix: Dict[str, Dict[str, float]] = {}
        for doc in docs:
            doc_chunks = doc.chunks
            doc_vec_idx = {c.chunk_id: i for i, c in enumerate(sub_chunks)}
            rows = [align_map[doc_vec_idx[c.chunk_id]] for c in doc_chunks]
            pair_map = pairwise_coverage(rows, doc_chunks, threshold)
            peer_matches = {f: pm for f, pm in pair_map.items() if pm.coverage > 0}
            matrix[doc.file_name] = {f: pm.coverage for f, pm in pair_map.items()}
            self.logger.info(
                f"[peer] {doc.file_name}: "
                + (
                    ", ".join(
                        f"{f}={pm.coverage:.0f}%"
                        for f, pm in sorted(peer_matches.items(), key=lambda kv: -kv[1].coverage)[:4]
                    )
                    if peer_matches
                    else "no matches"
                )
            )

            # 6. External similarity ---------------------------------------
            ext = self.reference.score(
                doc_chunks,
                sub_vecs[[doc_vec_idx[c.chunk_id] for c in doc_chunks]],
                ref_vecs,
                cfg["external"]["match_threshold"],
            )

            # 7. AI-content estimation -------------------------------------
            ai_signal = None
            if doc.full_text and len(doc.full_text.split()) >= 40:
                ai_signal = ai_score_fn(doc.full_text, [c.text for c in doc.chunks])
            else:
                from .ai_detector import AISignal

                ai_signal = AISignal(
                    score_0_100=0.0,
                    backend=ai_backend + " (insufficient text)",
                    features={},
                )

            bp_words = sum(c.n_words for c in doc_chunks if c.is_boilerplate)
            bp_chunks = sum(1 for c in doc_chunks if c.is_boilerplate)
            bp_summary = {
                "chunks": bp_chunks,
                "words": bp_words,
                "files_sharing_template": boilerplate_files,
            }

            analysis = DocAnalysis(
                doc=doc,
                peer_matches=peer_matches,
                external=ext,
                ai=ai_signal,
                boilerplate_summary=bp_summary,
                embedding_backend=self.embedding.backend,
                ai_backend=ai_backend,
            )
            analysis.scores = compute_scores(peer_matches, ext, ai_signal, cfg["risk"])
            top_pair = max(peer_matches.values(), key=lambda p: p.coverage) if peer_matches else None
            analysis.risk = compute_risk(analysis.scores, top_pair, cfg["risk"])
            analysis.explanation = compose_explanation(analysis, boilerplate_files)
            analyses.append(analysis)

        # 8. Reports ---------------------------------------------------------
        self._write_reports(analyses, matrix, output_dir)
        elapsed = time.time() - t_start
        self.logger.info(f"[pipeline] done: {len(docs)} files in {elapsed:.1f}s")
        return {
            "elapsed_s": elapsed,
            "files": [asdict(d) for d in docs],
            "analyses": analyses,
        }

    # ------------------------------------------------------------------ #
    def _write_reports(
        self,
        analyses: List[DocAnalysis],
        matrix: Dict[str, Dict[str, float]],
        output_dir: Path,
    ) -> None:
        cfg = self.cfg
        per_doc_dir = output_dir / "per_document"
        log_entries = []
        for a in analyses:
            slug = Path(a.doc.file_name).stem
            bp_files = a.boilerplate_summary.get("files_sharing_template", 0)
            write_json(per_doc_dir / f"{slug}.json", build_json_report(a, cfg, bp_files))
            md = build_markdown_report(a, bp_files)
            (per_doc_dir / f"{slug}.md").write_text(md, encoding="utf-8")
            log_entries.append(
                {
                    "file": a.doc.file_name,
                    "ok": a.doc.ok,
                    "error": a.doc.error,
                    "warnings": a.doc.warnings,
                    "chunks": len(a.doc.chunks),
                    "risk": a.risk,
                    "scores": a.scores,
                }
            )
        write_json(
            output_dir / "similarity_matrix.json",
            {
                "threshold": cfg["similarity"]["match_threshold"],
                "metric": "coverage: % of row document aligned to column document",
                "matrix": matrix,
            },
        )
        write_json(
            output_dir / "processing_log.json",
            {
                "config": cfg,
                "embedding_backend": self.embedding.backend if self.embedding else "n/a",
                "files": log_entries,
            },
        )
        summary_md = build_summary_markdown(analyses, matrix, cfg)
        (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")
        self.logger.info(
            f"[reports] wrote {len(analyses)} per-document reports + summary to {output_dir}"
        )
