import streamlit as st
from pathlib import Path
import subprocess
from src.graph import build_graph

st.set_page_config(page_title="Neura Dynamics — Weather + PDF RAG", page_icon="🤖", layout="centered")
st.title("Neura Dynamics — Weather + PDF RAG (LangGraph)")

with st.expander("1) Ingest PDF", expanded=True):
    st.write("Place your PDF at `data/sample.pdf` (a demo file works too).")
    if not Path("data/sample.pdf").exists():
        st.error("Missing `data/sample.pdf`. Please add the file.")
    if st.button("Ingest now", disabled=not Path("data/sample.pdf").exists()):
        out = subprocess.run(["python", "-m", "src.ingest"], capture_output=True, text=True)
        st.write("```bash\n" + (out.stdout or out.stderr) + "\n```")
        if out.returncode == 0:
            st.success("Ingested chunks into Qdrant (in-memory).")

st.divider()
st.caption("Ask things like *What's the weather in Bangalore?* or *Summarize the main ideas in the PDF.*")

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
