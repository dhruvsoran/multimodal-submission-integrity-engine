"""Vector index abstraction: FAISS when available, NumPy cosine otherwise.

For the 15-file prototype scale a NumPy dot-product over L2-normalized
embeddings IS an exact cosine-similarity index; FAISS (IndexFlatIP) is
auto-selected when installed to keep the production code path warm.  Both
backends return identical exact results on this scale.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)


class VectorIndex:
    """Exact inner-product index over normalized vectors (== cosine)."""

    def __init__(self, dim: int):
        self.dim = dim
        self._faiss = None
        self.backend = "numpy"
        try:
            import faiss  # type: ignore

            self._faiss = faiss.IndexFlatIP(dim)
            self.backend = "faiss"
            logger.info(f"[index] backend=faiss (IndexFlatIP, dim={dim})")
        except Exception:  # noqa: BLE001
            self._store = np.zeros((0, dim), dtype="float32")
            logger.info(f"[index] backend=numpy (exact cosine via dot product, dim={dim})")

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.size == 0:
            return
        if self._faiss is not None:
            self._faiss.add(vectors)
        else:
            self._store = np.vstack([self._store, vectors])

    @property
    def ntotal(self) -> int:
        return int(self._faiss.ntotal) if self._faiss is not None else len(self._store)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) of top-k nearest vectors per query row."""
        queries = np.asarray(queries, dtype="float32")
        if self.ntotal == 0:
            return np.zeros((len(queries), 0), "float32"), np.zeros((len(queries), 0), "int64")
        k = min(k, self.ntotal)
        if self._faiss is not None:
            scores, idx = self._faiss.search(queries, k)
            return scores, idx
        sims = queries @ self._store.T  # (n_q, n_all)
        k_eff = min(k, sims.shape[1])
        idx = np.argsort(-sims, axis=1)[:, :k_eff]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores, idx
