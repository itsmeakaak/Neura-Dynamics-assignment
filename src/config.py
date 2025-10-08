import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama  # chat interface

# Load .env early
load_dotenv(override=True)

def _resolve_path(p: str) -> str:
    """Return an absolute path; resolve relative to the repo root (../ from src/)."""
    if os.path.isabs(p):
        return p
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.normpath(os.path.join(repo_root, p))

# ---- Qdrant ----
def get_qdrant_client() -> QdrantClient:
    # Prefer an on-disk local store if QDRANT_PATH is set
    qdrant_path = os.getenv("QDRANT_PATH")
    if qdrant_path:
        qdrant_path = _resolve_path(qdrant_path)
        os.makedirs(qdrant_path, exist_ok=True)
        return QdrantClient(path=qdrant_path)

    # Otherwise fall back to memory or remote URL
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key)

    location = os.getenv("QDRANT_LOCATION", ":memory:")
    return QdrantClient(location=location)

# ---- Embeddings (local, free) ----
def get_embeddings():
    model = os.getenv("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model)

# ---- LLM (local, free via Ollama) ----
def get_llm():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")
    return ChatOllama(base_url=base_url, model=model, temperature=0)
