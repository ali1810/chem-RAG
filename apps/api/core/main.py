"""core/main.py — FastAPI application factory."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.config import get_settings
from routers import query, ingest, health

os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    logger.info("ChemRAG API starting | model={}", s.embedding_model)

    # Pre-load embedding model at startup
    from services.embedder import get_embedder
    get_embedder()
    logger.info("Embedding model loaded ✅")

    # Load existing FAISS index if it exists
    from services.vectorstore import get_vectorstore
    vs = get_vectorstore()
    doc_count = vs.count()
    logger.info("Vector store ready | docs={}", doc_count)

    yield
    logger.info("ChemRAG shutting down")


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title=s.app_title,
        version=s.app_version,
        description="🧪 RAG system for chemistry papers and PubChem abstracts",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(query.router,  prefix="/api/v1/query",  tags=["Query"])
    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])
    app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
    return app


app = create_app()
