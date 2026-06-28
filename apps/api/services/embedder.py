"""
services/embedder.py — Sentence-transformer embedding service.

Loads once at startup, encodes text into dense vectors.
Model: all-MiniLM-L6-v2 (fast, 384-dim, good for scientific text)
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from loguru import logger
from core.config import get_settings


class Embedder:
    """Wraps sentence-transformers for encode operations."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: {}", model_name)
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model ready | dim={}", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str] | str, batch_size: int = 32) -> np.ndarray:
        """Encode text(s) into normalised dense vectors."""
        if isinstance(texts, str):
            texts = [texts]
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # L2-normalise → cosine sim = dot product
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        return self.encode([query])[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Singleton embedder — loaded once at startup."""
    global _embedder
    if _embedder is None:
        s = get_settings()
        _embedder = Embedder(s.embedding_model)
    return _embedder
