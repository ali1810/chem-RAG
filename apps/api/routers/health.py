"""routers/health.py — Health check."""
from fastapi import APIRouter
from core.config import get_settings
from models.schemas import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health():
    s = get_settings()
    try:
        from services.vectorstore import get_vectorstore
        store = get_vectorstore()
        chunk_count = store.count()
        doc_count = len(store.get_documents())
        index_ready = chunk_count > 0
    except Exception:
        chunk_count = 0
        doc_count = 0
        index_ready = False

    return HealthResponse(
        status="ok",
        embedding_model=s.embedding_model,
        llm_model=s.groq_model,
        document_count=doc_count,
        chunk_count=chunk_count,
        index_ready=index_ready,
        groq_key_set=bool(s.groq_api_key),
    )
