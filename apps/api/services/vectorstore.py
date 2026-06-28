"""
services/vectorstore.py — FAISS vector store with JSON document store.

Architecture:
  - FAISS IndexFlatIP (inner product = cosine sim on normalised vectors)
  - JSON sidecar stores chunk metadata and text
  - Persistent: saves/loads from disk

Why FAISS:
  - In-process, no external service needed
  - Production-grade similarity search
  - Used by Meta, Google in production at scale
"""
from __future__ import annotations
import json
import os
import numpy as np
from pathlib import Path
from loguru import logger
from core.config import get_settings
from services.chunker import Chunk


class VectorStore:
    """FAISS-backed vector store with JSON document sidecar."""

    def __init__(self, index_path: str, dim: int = 384):
        import faiss
        self._dim = dim
        self._index_path = Path(index_path)
        self._meta_path = Path(index_path + "_meta.json")
        self._chunks: list[dict] = []

        # Load or create FAISS index
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path) as f:
                self._chunks = json.load(f)
            logger.info("Loaded FAISS index | vectors={} chunks={}",
                        self._index.ntotal, len(self._chunks))
        else:
            # Inner product index — works as cosine sim with normalised vectors
            self._index = faiss.IndexFlatIP(dim)
            logger.info("Created new FAISS index | dim={}", dim)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the store."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have same length")

        # Add to FAISS
        self._index.add(embeddings)

        # Store metadata
        for chunk in chunks:
            self._chunks.append({
                "chunk_id":    chunk.chunk_id,
                "doc_id":      chunk.doc_id,
                "title":       chunk.title,
                "source":      chunk.source,
                "text":        chunk.text,
                "chunk_index": chunk.chunk_index,
                "metadata":    chunk.metadata,
            })

        self._save()
        logger.debug("Added {} chunks | total={}", len(chunks), len(self._chunks))

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[dict, float]]:
        """
        Find top-k most similar chunks.
        Returns list of (chunk_metadata, score) tuples.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._chunks):
                results.append((self._chunks[idx], float(score)))

        return results

    def count(self) -> int:
        """Return number of indexed vectors."""
        return self._index.ntotal

    def doc_ids(self) -> set[str]:
        """Return set of all ingested document IDs."""
        return {c["doc_id"] for c in self._chunks}

    def get_documents(self) -> list[dict]:
        """Return document-level summaries."""
        docs: dict[str, dict] = {}
        for chunk in self._chunks:
            did = chunk["doc_id"]
            if did not in docs:
                docs[did] = {
                    "doc_id":      did,
                    "title":       chunk["title"],
                    "source":      chunk["source"],
                    "chunk_count": 0,
                    "preview":     chunk["text"][:200],
                }
            docs[did]["chunk_count"] += 1
        return list(docs.values())

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w") as f:
            json.dump(self._chunks, f)


_store: VectorStore | None = None


def get_vectorstore() -> VectorStore:
    """Singleton vector store."""
    global _store
    if _store is None:
        from services.embedder import get_embedder
        s = get_settings()
        dim = get_embedder().dim
        _store = VectorStore(s.index_path, dim=dim)
    return _store
