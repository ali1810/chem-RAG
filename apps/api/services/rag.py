"""
services/rag.py — Core RAG query pipeline.

Pipeline:
  User question
      → embed query
      → FAISS top-k retrieval
      → assemble context + prompt
      → LLM generation (Groq)
      → return answer + cited sources
"""
from __future__ import annotations
import os
from loguru import logger
from core.config import get_settings
from services.embedder import get_embedder
from services.vectorstore import get_vectorstore
from models.schemas import QueryResponse, SourceDocument


SYSTEM_PROMPT = """You are ChemRAG, an expert chemistry research assistant.
You answer questions about chemistry, drug discovery, molecular properties,
chemical reactions, and related scientific topics.

CRITICAL RULES:
1. Answer ONLY using the provided context documents.
2. If the context does not contain enough information, say so clearly.
3. Always cite which document(s) your answer comes from.
4. Do not invent facts, chemical data, or experimental values.
5. Be precise and scientific in your language.
6. If asked about a specific compound, report its properties as given in the context.
"""

QUERY_TEMPLATE = """Context documents:
{context}

---
Question: {question}

Answer based only on the context above. Cite the relevant document title(s) in your answer."""


def build_context(retrieved: list[tuple[dict, float]]) -> str:
    """Format retrieved chunks into context string."""
    parts = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        parts.append(
            f"[Document {i}: {chunk['title']}]\n"
            f"{chunk['text']}\n"
            f"(Source: {chunk['source']})"
        )
    return "\n\n".join(parts)


def query_rag(
    question: str,
    top_k: int = 5,
) -> QueryResponse:
    """
    Full RAG pipeline: retrieve → generate → return.

    Args:
        question: Natural language chemistry question
        top_k: Number of chunks to retrieve

    Returns:
        QueryResponse with answer, sources, and metadata
    """
    s = get_settings()
    embedder = get_embedder()
    store    = get_vectorstore()

    # ── Step 1: Embed query ───────────────────────────────────────────────────
    logger.debug("Embedding query: {}", question[:80])
    query_vec = embedder.encode_query(question)

    # ── Step 2: Retrieve top-k chunks ─────────────────────────────────────────
    retrieved = store.search(query_vec, top_k=top_k)
    logger.debug("Retrieved {} chunks", len(retrieved))

    if not retrieved:
        return QueryResponse(
            question=question,
            answer="No documents have been ingested yet. Please ingest chemistry documents first using the Ingest tab.",
            sources=[],
            model_used=s.groq_model,
            retrieval_count=0,
            grounded=False,
        )

    # ── Step 3: Build context ─────────────────────────────────────────────────
    context = build_context(retrieved)
    prompt  = QUERY_TEMPLATE.format(context=context, question=question)

    # ── Step 4: Generate answer ───────────────────────────────────────────────
    answer = _generate(prompt, s)

    # ── Step 5: Build source list ─────────────────────────────────────────────
    sources = []
    seen_docs: set[str] = set()
    for chunk, score in retrieved:
        doc_key = chunk["doc_id"]
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            sources.append(SourceDocument(
                title=chunk["title"],
                source=chunk["source"],
                text_excerpt=chunk["text"][:300] + "...",
                score=round(score, 4),
                chunk_id=chunk["chunk_id"],
            ))

    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        model_used=s.groq_model,
        retrieval_count=len(retrieved),
        grounded=True,
    )


def _generate(prompt: str, s) -> str:
    """Call Groq LLM for generation."""
    if not s.groq_api_key:
        # Fallback: return context summary without LLM
        logger.warning("No GROQ_API_KEY — returning retrieval-only response")
        return (
            "⚠️ LLM generation unavailable (no GROQ_API_KEY set). "
            "Retrieved documents are shown below. "
            "Add your Groq API key to .env to enable full RAG generation."
        )
    try:
        from groq import Groq
        client = Groq(api_key=s.groq_api_key)
        response = client.chat.completions.create(
            model=s.groq_model,
            temperature=0.1,      # low temp for factual chemistry
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("LLM generation failed: {}", e)
        return f"Generation error: {e}. Retrieved documents are shown below."
