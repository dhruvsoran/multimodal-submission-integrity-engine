"""Embedding backend with graceful degradation.

Primary: sentence-transformers ``all-MiniLM-L6-v2`` (384-dim, CPU-friendly,
the de-facto open-source semantic similarity model named in the brief).
Fallback: scikit-learn TF-IDF vectors, so the pipeline still runs end-to-end
on machines where the neural model cannot be downloaded.  The active backend
is reported in every output so results are never silently degraded.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEngine:
    """Computes L2-normalized embeddings (cosine == dot product)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None):
        self.model_name = model_name
        self.backend = "sentence-transformers"
        self._model = None
        self._tfidf = None
        self._warned = False
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": "cpu"}
            if cache_dir:
                kwargs["cache_folder"] = cache_dir
            self._model = SentenceTransformer(model_name, **kwargs)
            get_dim = getattr(
                self._model,
                "get_sentence_embedding_dimension",
                None,
            ) or getattr(self._model, "get_embedding_dimension")
            self.dim = int(get_dim())
            logger.info(
                f"[embeddings] backend=sentence-transformers model={model_name} dim={self.dim}"
            )
        except Exception as exc:  # noqa: BLE001
            self.backend = "tfidf-fallback"
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(
                lowercase=True, stop_words="english", sublinear_tf=True, min_df=1
            )
            self.dim = 0  # determined on first fit
            self._load_error = str(exc)
            logger.warning(
                "[embeddings] sentence-transformers unavailable "
                f"({exc.__class__.__name__}); using TF-IDF lexical fallback. "
                "Install torch + sentence-transformers for semantic embeddings."
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts; rows are L2-normalized."""
        if not texts:
            return np.zeros((0, self.dim or 1), dtype="float32")
        if self.backend == "sentence-transformers":
            vecs = self._model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(vecs, dtype="float32")
        # TF-IDF fallback: fit on the incoming batch (the pipeline embeds the
        # whole corpus in one call, which keeps the matrix consistent).
        matrix = self._tfidf.fit_transform(texts)
        self.dim = matrix.shape[1]
        from sklearn.preprocessing import normalize

        if not self._warned:
            self._warned = True
        return normalize(matrix).astype("float32")

    def embed_query_batch(self, texts: List[str]) -> np.ndarray:
        """Alias kept for clarity when embedding after a corpus-wide fit."""
        if self.backend == "tfidf-fallback":
            matrix = self._tfidf.transform(texts)
            from sklearn.preprocessing import normalize

            return normalize(matrix).astype("float32")
        return self.embed(texts)
