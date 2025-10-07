from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from .config import get_qdrant_client, get_embeddings
import os

COLLECTION = os.getenv("QDRANT_COLLECTION", "nd_pdf_collection")

def load_and_split(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))  # PDF → Documents
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(docs)

def build_store(client: QdrantClient, chunks: List):
    embeddings = get_embeddings()  # local HF embeddings
    return QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION,
    )

if __name__ == "__main__":
    pdf = Path("data/sample.pdf")
    assert pdf.exists(), "Put your PDF at data/sample.pdf"
    chunks = load_and_split(pdf)
    client = get_qdrant_client()
    build_store(client, chunks)
    print(f"Ingested {len(chunks)} chunks into '{COLLECTION}'.")
