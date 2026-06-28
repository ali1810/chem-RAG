"""api_client.py — HTTP client for ChemRAG FastAPI backend."""
import httpx

BASE = "http://127.0.0.1:8000/api/v1"
TIMEOUT = 60


def query(question: str, top_k: int = 5) -> dict:
    with httpx.Client(timeout=TIMEOUT) as c:
        r = c.post(f"{BASE}/query/", json={"question": question, "top_k": top_k})
        r.raise_for_status()
        return r.json()


def get_examples() -> list[str]:
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{BASE}/query/examples")
        r.raise_for_status()
        return r.json()["examples"]


def ingest_samples() -> dict:
    with httpx.Client(timeout=120) as c:
        r = c.post(f"{BASE}/ingest/samples")
        r.raise_for_status()
        return r.json()


def ingest_pubchem(cids: list[int]) -> dict:
    with httpx.Client(timeout=120) as c:
        r = c.post(f"{BASE}/ingest/pubchem", json={"cids": cids})
        r.raise_for_status()
        return r.json()


def ingest_text(title: str, text: str, source: str = "manual") -> dict:
    with httpx.Client(timeout=30) as c:
        r = c.post(f"{BASE}/ingest/text",
                   json={"title": title, "text": text, "source": source})
        r.raise_for_status()
        return r.json()


def list_documents() -> list[dict]:
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{BASE}/ingest/documents")
        r.raise_for_status()
        return r.json()


def get_health() -> dict:
    with httpx.Client(timeout=5) as c:
        r = c.get(f"{BASE}/health/")
        r.raise_for_status()
        return r.json()
