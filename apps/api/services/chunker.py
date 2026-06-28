"""
services/chunker.py — Text chunking for RAG ingestion.

Splits documents into overlapping chunks for embedding.
Uses sentence-aware splitting to avoid cutting mid-sentence.
"""
from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field
from core.config import get_settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Split on . ! ? followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    Tries to break at sentence boundaries.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_words: list[str] = []
    current_size = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if current_size + word_count > chunk_size and current_words:
            # Save current chunk
            chunks.append(" ".join(current_words))
            # Keep overlap
            overlap_words = current_words[-chunk_overlap:] if chunk_overlap > 0 else []
            current_words = overlap_words + words
            current_size = len(current_words)
        else:
            current_words.extend(words)
            current_size += word_count

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    source: str,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Chunk a document into overlapping text segments."""
    s = get_settings()
    raw_chunks = chunk_text(text, s.chunk_size, s.chunk_overlap)

    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            title=title,
            source=source,
            text=chunk,
            chunk_index=i,
            metadata=metadata or {},
        )
        for i, chunk in enumerate(raw_chunks)
    ]
