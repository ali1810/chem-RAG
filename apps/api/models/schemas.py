"""models/schemas.py — Pydantic schemas for all endpoints."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestPubChemRequest(BaseModel):
    cids: list[int] = Field(
        default=[2244, 5090, 4091, 3672, 60961, 123601, 5353499],
        description="PubChem Compound IDs to ingest",
        example=[2244, 5090, 4091]
    )

class IngestTextRequest(BaseModel):
    title: str
    text: str
    source: str = "manual"
    metadata: dict = Field(default_factory=dict)

class IngestResponse(BaseModel):
    ingested: int
    total_chunks: int
    total_documents: int
    message: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        example="What is the solubility of aspirin in water?",
        description="Natural language chemistry question"
    )
    top_k: int = Field(default=5, ge=1, le=10)
    include_sources: bool = True

class SourceDocument(BaseModel):
    title: str
    source: str
    text_excerpt: str
    score: float
    chunk_id: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    model_used: str
    retrieval_count: int
    grounded: bool           # True if answer is based on retrieved context


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    source: str
    chunk_count: int
    preview: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    llm_model: str
    document_count: int
    chunk_count: int
    index_ready: bool
    groq_key_set: bool
