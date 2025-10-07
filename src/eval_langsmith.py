"""
Trace a couple of runs to LangSmith (no paid judge models).
Requires LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env (free dev seat).
"""
import os
from src.graph import build_graph

def run_examples():
    g = build_graph()
    examples = [
        {"question": "What's the weather in Delhi?"},
        {"question": "Summarize the main ideas of the PDF."},
    ]
    for ex in examples:
        out = g.invoke(ex)
        print(f"Q: {ex['question']}\nA: {out['answer']}\n")

if __name__ == "__main__":
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() not in ("true", "1", "yes"):
        print("Tip: enable LangSmith tracing by setting LANGCHAIN_TRACING_V2=true in your .env")
    run_examples()
