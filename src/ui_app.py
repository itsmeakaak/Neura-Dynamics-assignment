import streamlit as st
from pathlib import Path
import subprocess, sys, os

# NEW: load .env so Streamlit sees your keys
from dotenv import load_dotenv
load_dotenv(override=True)  # ensure values from .env are available

# Robust import whether CWD is repo root or src/
try:
    from src.graph import build_graph
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))                  # add src/
    sys.path.append(os.path.dirname(os.path.dirname(__file__))) # add project root
    try:
        from src.graph import build_graph
    except ModuleNotFoundError:
        from graph import build_graph  # fallback if launched from src/

st.set_page_config(page_title="Neura Dynamics — Weather + PDF RAG", page_icon="🤖", layout="centered")
st.title("Neura Dynamics — Weather + PDF RAG (LangGraph)")

with st.expander("1) Ingest PDF", expanded=True):
    st.write("Place your PDF at `data/sample.pdf` (a demo file works too).")
    pdf_ok = Path("data/sample.pdf").exists()
    if not pdf_ok:
        st.error("Missing `data/sample.pdf`. Please add the file.")
    if st.button("Ingest now", disabled=not pdf_ok):
        out = subprocess.run([sys.executable, "-m", "src.ingest"], capture_output=True, text=True)
        st.write("```bash\n" + (out.stdout or out.stderr) + "\n```")
        if out.returncode == 0:
            st.success("Ingested chunks into Qdrant (in-memory).")

st.divider()
st.caption("Ask things like *What's the weather in Bengaluru?* or *Summarize the main ideas in the PDF.*")

if "chat" not in st.session_state:
    st.session_state["chat"] = []

# render history
for who, msg in st.session_state["chat"]:
    with st.chat_message(who):
        st.markdown(msg)

user_msg = st.chat_input("Type your question…")
if user_msg:
    st.session_state["chat"].append(("user", user_msg))
    with st.chat_message("assistant"):
        try:
            graph = build_graph()
            result = graph.invoke({"question": user_msg})
            answer = result["answer"]
        except Exception as e:
            answer = f"Error: {e}"
        st.markdown(answer)
        st.session_state["chat"].append(("assistant", answer))
