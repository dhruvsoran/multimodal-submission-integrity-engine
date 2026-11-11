#!/usr/bin/env python3
"""Single-command entry point for the Submission Integrity Engine.

Usage (as required by the assignment):

    python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports

Outputs:
    <output_dir>/per_document/<file>.json + .md   audit report per submission
    <output_dir>/summary.md                       batch summary + matrix
    <output_dir>/similarity_matrix.json           pairwise coverage matrix
    <output_dir>/processing_log.json              config + per-file log
"""

from __future__ import annotations

import os

# Quiet the HF hub symlink warning on Windows (cosmetic noise on every run).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from integrity.pipeline import DEFAULT_CONFIG, IntegrityPipeline, deep_merge
from integrity.utils import get_logger

logger = get_logger("cli")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_integrity_check.py",
        description=(
            "Multimodal Submission Integrity & Plagiarism Detection Engine - "
            "ingests PDF/DOCX/MP4 submissions and emits explainable integrity reports."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input_dir", required=True, help="Folder containing candidate submissions.")
    ap.add_argument("--output_dir", default="./reports", help="Folder for generated reports.")
    ap.add_argument(
        "--config",
        default=str(ROOT / "config" / "config.json"),
        help="Optional JSON config overriding the defaults.",
    )
    ap.add_argument(
        "--whisper_model",
        default=None,
        choices=["tiny", "base", "small"],
        help="Whisper model size (overrides config).",
    )
    ap.add_argument(
        "--ai_backend",
        default=None,
        choices=["heuristic", "hf"],
        help="AI-content detector backend: reproducible heuristic or HuggingFace RoBERTa.",
    )
    ap.add_argument(
        "--embedding_model",
        default=None,
        help="Sentence-Transformers model id (overrides config).",
    )
    ap.add_argument(
        "--match_threshold",
        type=float,
        default=None,
        help="Cosine similarity above which a chunk pair counts as matched.",
    )
    ap.add_argument(
        "--fetch-web",
        action="store_true",
        help="Extend the local reference KB via the free Wikipedia API at runtime.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return 2

    cfg: dict = dict(DEFAULT_CONFIG)
    config_path = Path(args.config)
    if config_path.exists():
        try:
            cfg = deep_merge(DEFAULT_CONFIG, json.loads(config_path.read_text(encoding="utf-8")))
            logger.info(f"[cli] loaded config: {config_path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[cli] could not parse {config_path} ({exc}); using defaults.")
    else:
        logger.info("[cli] no config file found; using built-in defaults.")

    # CLI overrides.
    if args.whisper_model:
        cfg["whisper_model"] = args.whisper_model
    if args.ai_backend:
        cfg["ai"]["backend"] = args.ai_backend
    if args.embedding_model:
        cfg["embedding_model"] = args.embedding_model
    if args.match_threshold:
        cfg["similarity"]["match_threshold"] = args.match_threshold
    if args.fetch_web:
        from integrity.external import WIKIPEDIA_TOPICS, fetch_wikipedia_pages

        ref_dir = Path(ROOT) / cfg["external"]["reference_dir"]
        saved = fetch_wikipedia_pages(WIKIPEDIA_TOPICS, ref_dir / "online")
        logger.info(f"[cli] web fetch: {len(saved)} pages added to the reference KB")

    pipeline = IntegrityPipeline(cfg, log_file=Path(args.output_dir) / "pipeline.log")
    result = pipeline.run(input_dir, args.output_dir)

    n_flagged = 0
    for analysis in result["analyses"]:
        flag = "" if analysis.risk == "LOW" else f"  <-- {analysis.risk}"
        if analysis.risk != "LOW":
            n_flagged += 1
        print(
            f"  {analysis.doc.file_name:<28} peer "
            f"{analysis.scores.get('peer_similarity_percentage', 0):5.1f}%   external "
            f"{analysis.scores.get('external_similarity_percentage', 0):5.1f}%   AI "
            f"{analysis.scores.get('ai_generated_probability', 0):5.1f}%   "
            f"[{analysis.risk}]{flag}"
        )
    print(
        f"\nReports written to: {Path(args.output_dir).resolve()}\n"
        f"Flagged submissions: {n_flagged}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
