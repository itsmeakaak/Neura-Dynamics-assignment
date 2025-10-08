import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama  # chat interface

# Load .env so anything that imports config sees your keys
load_dotenv(override=True)

# ---- Qdrant ----
def get_qdrant_client() -> QdrantClient:
    # 1) If remote URL provided, use it
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key)

    # 2) Prefer local persistent path if provided
    qdrant_path = os.getenv("QDRANT_PATH")
    if qdrant_path:
        os.makedirs(qdrant_path, exist_ok=True)
        return QdrantClient(path=qdrant_path)

    # 3) Fallback: previous behavior (in-memory)
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
