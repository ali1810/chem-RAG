"""routers/query.py — RAG query endpoint."""
from fastapi import APIRouter, HTTPException
from loguru import logger
from models.schemas import QueryRequest, QueryResponse
from services.rag import query_rag

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a chemistry question answered using RAG.
    Retrieves relevant document chunks and generates a grounded answer.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    logger.info("Query: {}", question[:100])
    try:
        result = query_rag(question, top_k=request.top_k)
        return result
    except Exception as e:
        logger.error("Query failed: {}", e)
        raise HTTPException(500, f"Query failed: {e}")


@router.get("/examples")
async def get_example_questions():
    """Return example questions for the demo."""
    return {"examples": [
        "What is the aqueous solubility of aspirin?",
        "How does molecular weight affect drug solubility?",
        "What is retrosynthesis and how does it work?",
        "Explain how ChemBERT represents chemical structures",
        "What is RAG and how does it prevent hallucination?",
        "What reaction classes are in the USPTO-50K dataset?",
        "How does SMILES augmentation improve model performance?",
        "What is the role of LogP in drug discovery?",
        "How does FAISS perform vector similarity search?",
        "What is the difference between BERT and GPT architectures?",
    ]}
