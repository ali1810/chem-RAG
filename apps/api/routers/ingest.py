"""routers/ingest.py — Document ingestion endpoints."""
from fastapi import APIRouter, HTTPException
from loguru import logger
from models.schemas import (
    IngestPubChemRequest, IngestTextRequest, IngestResponse
)
from services.ingest import (
    ingest_pubchem_cids, ingest_document, ingest_sample_documents
)
from services.vectorstore import get_vectorstore
from models.schemas import DocumentSummary

router = APIRouter()


@router.post("/pubchem", response_model=IngestResponse)
async def ingest_pubchem(request: IngestPubChemRequest):
    """
    Ingest chemistry compounds from PubChem by CID.
    Fetches descriptions and properties, chunks, embeds, and indexes.
    """
    if not request.cids:
        raise HTTPException(400, "No CIDs provided")
    if len(request.cids) > 50:
        raise HTTPException(400, "Maximum 50 CIDs per request")

    logger.info("Ingesting {} PubChem CIDs", len(request.cids))
    try:
        result = ingest_pubchem_cids(request.cids)
        return IngestResponse(
            ingested=result["ingested"],
            total_chunks=result["total_chunks"],
            total_documents=result["total_documents"],
            message=f"Successfully ingested {result['ingested']} compounds ({result['total_chunks']} chunks)"
        )
    except Exception as e:
        logger.error("PubChem ingestion failed: {}", e)
        raise HTTPException(500, f"Ingestion failed: {e}")


@router.post("/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest):
    """Ingest a raw text document (paper, abstract, notes)."""
    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    import uuid
    doc_id = f"manual_{uuid.uuid4().hex[:8]}"

    try:
        chunks = ingest_document(
            doc_id=doc_id,
            title=request.title,
            text=request.text,
            source=request.source,
            metadata=request.metadata,
        )
        store = get_vectorstore()
        return IngestResponse(
            ingested=1 if chunks else 0,
            total_chunks=len(chunks),
            total_documents=store.count(),
            message=f"Ingested '{request.title}' ({len(chunks)} chunks)"
        )
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")


@router.post("/samples", response_model=IngestResponse)
async def ingest_samples():
    """
    Ingest built-in sample chemistry documents.
    Includes: solubility, retrosynthesis, NER, transformers, RAG.
    """
    logger.info("Ingesting sample chemistry documents")
    try:
        result = ingest_sample_documents()
        return IngestResponse(
            ingested=result["ingested"],
            total_chunks=result["total_chunks"],
            total_documents=result["total_documents"],
            message=f"Ingested {result['ingested']} sample documents ({result['total_chunks']} chunks)"
        )
    except Exception as e:
        raise HTTPException(500, f"Sample ingestion failed: {e}")


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents():
    """List all ingested documents."""
    store = get_vectorstore()
    return [DocumentSummary(**d) for d in store.get_documents()]
