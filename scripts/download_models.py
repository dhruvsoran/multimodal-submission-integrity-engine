#!/usr/bin/env python3
"""Pre-download all models used by the pipeline (optional but recommended).

Running this once means the main pipeline works offline afterwards.

    python scripts/download_models.py            # embedding + whisper
    python scripts/download_models.py --ai-hf    # + optional HF AI-detector
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from integrity.utils import get_logger

logger = get_logger("download")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-hf", action="store_true", help="Also fetch the HF AI detector.")
    ap.add_argument("--whisper", default="base", choices=["tiny", "base", "small"])
    args = ap.parse_args()

    logger.info("Downloading embedding model: sentence-transformers/all-MiniLM-L6-v2")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    logger.info("Embedding model cached.")

    logger.info(f"Downloading Whisper model: {args.whisper} (faster-whisper / int8)")
    from faster_whisper import WhisperModel

    WhisperModel(args.whisper, device="cpu", compute_type="int8")
    logger.info("Whisper cached.")

    if args.ai_hf:
        logger.info("Downloading HF AI detector: Hello-SimpleAI/chatgpt-detector-roberta")
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        mid = "Hello-SimpleAI/chatgpt-detector-roberta"
        AutoTokenizer.from_pretrained(mid)
        AutoModelForSequenceClassification.from_pretrained(mid)
        logger.info("HF AI detector cached.")

    logger.info("All requested models are cached locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
