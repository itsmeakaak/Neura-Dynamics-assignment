import streamlit as st
from pathlib import Path
import subprocess, sys, os

# Robust import: works whether Streamlit's CWD is repo root or src/
# (Python searches the script dir first; we add repo root if needed.)
# Docs: sys.path init (first entry = script dir). 
# Streamlit chat APIs used below.
# Refs: python docs on sys.path; Streamlit chat docs.
try:
    from src.graph import build_graph
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(__file__))                  # add src/
    sys.path.append(os.path.dirname(os.path.dirname(__file__))) # add project root
    try:
        from src.graph import build_graph
    except ModuleNotFoundError:
        from graph import build_graph  # fallback if launched from src/
# sys.path behavior: :contentReference[oaicite:0]{index=0}
# st.chat_message / st.chat_input: :contentReference[oaicite:1]{index=1}

st.set_page_config(page_title="Neura Dynamics — Weather + PDF RAG", page_icon="🤖", layout="centered")
st.title("Neura Dynamics — Weather + PDF RAG (LangGraph)")

with st.expander("1) Ingest PDF", expanded=True):
    st.write("Place your PDF at `data/sample.pdf` (a demo file works too).")
    pdf_ok = Path("data/sample.pdf").exists()
    if not pdf_ok:
        st.error("Missing `data/sample.pdf`. Please add the file.")
    if st.button("Ingest now", disabled=not pdf_ok):
        # Use the current Python interpreter to avoid venv/path issues
        cmd = [sys.executable, "-m", "src.ingest"]
        out = subprocess.run(cmd, capture_output=True, text=True)
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
