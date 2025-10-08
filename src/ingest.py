import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from .config import get_qdrant_client, get_embeddings

COLLECTION = "nd_pdf_collection"
PDF_PATH = "data/sample.pdf"

def load_chunks(path=PDF_PATH):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

def ensure_collection(client, dim=384):
    # Create if missing (works for local persistent or :memory:)
    try:
        client.get_collection(COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def build_store(chunks):
    client = get_qdrant_client()
    embeddings = get_embeddings()
    ensure_collection(client, dim=384)
    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )
    store.add_documents(chunks)
    return store

if __name__ == "__main__":
    load_dotenv(override=True)
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing {PDF_PATH}. Add a PDF and retry.")
    chunks = load_chunks()
    build_store(chunks)
    print(f"Ingested {len(chunks)} chunks into '{COLLECTION}'")
