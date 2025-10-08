import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

COLLECTION = "nd_pdf_collection"
PDF_PATH = "data/sample.pdf"

def load_chunks(path=PDF_PATH):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

def get_embeddings():
    model_name = os.getenv("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)

def build_store(chunks):
    # true local, zero-cost store (no server)
    location = os.getenv("QDRANT_LOCATION", ":memory:")
    client = QdrantClient(location=location)  # ":memory:" = in-memory mode. :contentReference[oaicite:1]{index=1}
    embeddings = get_embeddings()
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    store.add_documents(chunks)
    return store

if __name__ == "__main__":
    load_dotenv()
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing {PDF_PATH}. Add a PDF and retry.")
    chunks = load_chunks()
    build_store(chunks)
    print(f"Ingested {len(chunks)} chunks into '{COLLECTION}'")
