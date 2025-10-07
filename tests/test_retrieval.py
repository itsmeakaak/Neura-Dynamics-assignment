from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from src.config import get_embeddings

def test_retriever_returns_docs():
    client = QdrantClient(location=":memory:")
    embed = get_embeddings()
    vs = QdrantVectorStore.from_texts(
        ["LangGraph routes tools.", "Qdrant stores embeddings."],
        embedding=embed,
        client=client,
        collection_name="test",
    )
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("What does Qdrant do?")
    assert len(docs) >= 1
