# 🧪 ChemRAG — Chemical Literature RAG Assistant

A production-grade Retrieval-Augmented Generation (RAG) system for chemistry papers and PubChem abstracts.

Built as part of the ChemPredict platform by **Dr. Mushtaq Ali, KIT**.

## Architecture

```
User Query
    │
    ▼
Streamlit Frontend (app.py)
    │  HTTP
    ▼
FastAPI Backend (main.py)
    │
    ├──→ [1] Query Embedding (sentence-transformers)
    │
    ├──→ [2] Vector Retrieval (FAISS index)
    │          └── Top-k chemistry chunks
    │
    ├──→ [3] Context Assembly + Prompt Engineering
    │
    └──→ [4] LLM Generation (Groq llama-3.1-8b-instant)
               └── Grounded answer + citations
```

## Components

- **Ingestion pipeline** — fetches PubChem abstracts + chemistry papers, chunks, embeds, indexes
- **Vector store** — FAISS index with sentence-transformers embeddings
- **FastAPI backend** — `/ingest`, `/query`, `/documents`, `/health` endpoints
- **Streamlit frontend** — clean chemistry-themed UI for demo

## Quick Start

```bash
# 1. Set up API
cd apps/api
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY

# 2. Ingest sample chemistry documents
python -m services.ingest

# 3. Start API
uvicorn core.main:app --reload

# 4. Start Streamlit (new terminal)
cd apps/streamlit
pip install -r requirements.txt
streamlit run app.py
```
