from typing import Dict, Any
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .config import get_qdrant_client, get_embeddings, get_llm
import os

COLLECTION = os.getenv("QDRANT_COLLECTION", "nd_pdf_collection")

def get_vectorstore(client: QdrantClient) -> QdrantVectorStore:
    embeddings = get_embeddings()  # local HF embeddings
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

def make_rag_chain(vs: QdrantVectorStore):
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You answer using ONLY the provided context. Be concise."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    llm = get_llm()
    parser = StrOutputParser()

    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = {
        "context": retriever | _format_docs,
        "question": RunnablePassthrough(),
    } | prompt | llm | parser

    return chain

def answer_from_pdf(question: str) -> str:
    client = get_qdrant_client()
    vs = get_vectorstore(client)
    chain = make_rag_chain(vs)
    return chain.invoke(question)
