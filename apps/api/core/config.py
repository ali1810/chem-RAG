"""core/config.py — Application configuration."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        case_sensitive=False, extra="ignore"
    )
    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG settings
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 5

    # Storage
    index_path: str = "./data/faiss_index"
    docs_path: str = "./data/documents.json"

    # App
    app_title: str = "ChemRAG"
    app_version: str = "1.0.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
