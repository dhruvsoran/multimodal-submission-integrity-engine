"""AI-content estimation.

Two interchangeable backends:

1. ``heuristic`` (default): a fully reproducible, offline statistical
   detector.  LLM prose tends to be "flat": uniform sentence lengths (low
   burstiness), heavy discourse-marker use, no contractions, polished
   punctuation (semicolons, em dashes) and near-zero first-person voice.
   These proxies are combined into a calibrated 0-100 score.

2. ``hf`` (optional): a Hugging Face RoBERTa AI-detector
   (``Hello-SimpleAI/chatgpt-detector-roberta``).  Requires the model
   download; falls back to the heuristic with a warning when offline.

Both paths are wrapped so a failure degrades gracefully instead of
crashing the pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .utils import get_logger

logger = get_logger(__name__)

TRANSITIONS = [
    "furthermore", "moreover", "in addition", "additionally", "consequently",
    "therefore", "thus", "in conclusion", "overall", "notably", "importantly",
    "specifically", "however", "on the other hand", "in summary", "further",
    "as a result", "it is important to note", "plays a crucial role",
    "a testament to", "in today's world", "landscape", "foster", "leverage",
]

CONTRACTIONS = re.compile(
    r"\b(?:don't|doesn't|didn't|can't|couldn't|won't|wouldn't|shouldn't|isn't|"
    r"aren't|wasn't|weren't|it's|that's|there's|we're|they're|i'm|we've|"
    r"we'd|we'll|i've|you're|he's|she's|let's)\b",
    re.I,
)
FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|we've|we're|we'd|my team|i'd|i've)\b", re.I)
POLISHED_PUNCT = re.compile(r"[;—–]|\s-\s")


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _squash(x: float, k: float = 6.0) -> float:
    """Logistic squashing so mid-range evidence still lands decisively."""
    return 1.0 / (1.0 + math.exp(-k * (x - 0.5)))


@dataclass
class AISignal:
    score_0_100: float
    backend: str
    features: Dict[str, float]
    note: Optional[str] = None


class HeuristicAIDetector:
    """Burstiness / style-composite AI detector (reproducible, no downloads).

    Feature weights were tuned on the synthetic ground-truth corpus
    (LLM-style submission vs. independently written ones) and are documented
    in EVALUATION.md.
    """

    WEIGHTS = {
        "low_burstiness": 0.30,
        "transition_density": 0.25,
        "no_contractions": 0.15,
        "polished_punctuation": 0.15,
        "low_first_person": 0.15,
    }

    def score_document(self, text: str, chunks: Optional[List[str]] = None) -> AISignal:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3]
        words = text.split()
        n_words = max(1, len(words))
        n_sents = max(1, len(sentences))

        # 1. Burstiness: LLM text has unusually uniform sentence lengths.
        lens = [len(s.split()) for s in sentences]
        mean_len = sum(lens) / n_sents
        if len(lens) > 1 and mean_len > 0:
            stdev = (sum((l - mean_len) ** 2 for l in lens) / (len(lens) - 1)) ** 0.5
            cv = stdev / mean_len
        else:
            cv = 0.5
        low_burst = _clamp01((0.55 - cv) / 0.30)

        # 2. Discourse-marker density per 100 words (LLM overuse).
        lowered = text.lower()
        n_trans = sum(lowered.count(t) for t in TRANSITIONS)
        trans_density = 100.0 * n_trans / n_words
        trans_score = _clamp01(trans_density / 6.0)

        # 3. Contraction absence (LLM formal register).
        n_contr = len(CONTRACTIONS.findall(text))
        no_contr_score = _clamp01(1.0 - n_contr / max(1.0, n_words / 60.0))

        # 4. Polished punctuation (semicolons / dashes per 100 words).
        n_polish = len(POLISHED_PUNCT.findall(text))
        polish_score = _clamp01(n_polish / max(1.0, n_words) * 100.0 / 3.0)

        # 5. First-person voice absence.
        n_first = len(FIRST_PERSON.findall(text))
        first_person_score = _clamp01(1.0 - n_first / max(1.0, n_words / 120.0))

        features = {
            "sentence_length_cv": round(cv, 3),
            "transitions_per_100_words": round(trans_density, 2),
            "contractions": n_contr,
            "polished_punct_per_100w": round(100.0 * n_polish / n_words, 2),
            "first_person_hits": n_first,
        }
        raw = (
            self.WEIGHTS["low_burstiness"] * low_burst
            + self.WEIGHTS["transition_density"] * trans_score
            + self.WEIGHTS["no_contractions"] * no_contr_score
            + self.WEIGHTS["polished_punctuation"] * polish_score
            + self.WEIGHTS["low_first_person"] * first_person_score
        )
        prob = _squash(raw)
        return AISignal(
            score_0_100=round(100.0 * prob, 1),
            backend="heuristic-burstiness",
            features=features,
        )


class HFAIDetector:
    """Optional Hugging Face RoBERTa AI-detector backend."""

    MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta"

    def __init__(self):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        self._tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._clf = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID),
            tokenizer=self._tok,
            truncation=True,
            max_length=512,
        )
        logger.info(f"[ai] HF detector loaded: {self.MODEL_ID}")

    def score_document(self, text: str, chunks: Optional[List[str]] = None) -> AISignal:
        texts = chunks if chunks else [text]
        # Score up to 8 spans to keep CPU latency bounded.
        spans = [t[:1500] for t in texts[:8]]
        probs: List[float] = []
        labels: List[str] = []
        for span in spans:
            out = self._clf(span)[0]
            labels.append(out["label"])
            # chatgpt-detector-roberta: label "chatgpt" (or 1) = machine text.
            if str(out["label"]).lower() in {"chatgpt", "1", "label_1", "machine"}:
                probs.append(float(out["score"]))
            else:
                probs.append(1.0 - float(out["score"]))
        prob = sum(probs) / len(probs) if probs else 0.0
        return AISignal(
            score_0_100=round(100.0 * prob, 1),
            backend=f"hf:{self.MODEL_ID}",
            features={"spans_scored": len(probs), "labels": labels},
        )


def build_ai_detector(backend: str = "heuristic"):
    """Factory; never raises - falls back to the heuristic on load failure."""
    if backend == "hf":
        try:
            hf = HFAIDetector()
            return lambda text, chunks=None: hf.score_document(text, chunks), "hf"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"[ai] HF detector unavailable ({exc.__class__.__name__}: {exc}); "
                "falling back to heuristic detector."
            )
    detector = HeuristicAIDetector()
    return detector.score_document, "heuristic-burstiness"
