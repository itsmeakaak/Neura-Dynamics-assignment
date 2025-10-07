# Neura-Dynamics-assignment

Minimal AI pipeline with **LangGraph + LangChain + LangSmith**, routing between:
- **Weather** (OpenWeatherMap) and
- **PDF RAG** (Qdrant vector DB).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill keys
