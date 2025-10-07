import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

load_dotenv()

# Qdrant
QDRANT_LOCATION = os.getenv("QDRANT_LOCATION", ":memory:")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def get_qdrant_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(location=QDRANT_LOCATION, api_key=QDRANT_API_KEY)
    return QdrantClient(location=QDRANT_LOCATION)

# Embeddings (local, free)
def get_embeddings():
    model = os.getenv("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model)

# LLM (local, free via Ollama)
def get_llm():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    return OllamaLLM(base_url=base_url, model=model, temperature=0)
