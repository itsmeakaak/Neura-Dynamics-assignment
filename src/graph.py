from typing import TypedDict, Literal, Optional
import re
from dotenv import load_dotenv
load_dotenv(override=True)  # make sure env is present when graph imports

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .weather import get_weather
from .rag import answer_from_pdf
from .config import get_llm

Route = Literal["weather", "rag"]

class State(TypedDict, total=False):
    question: str
    route: Route
    city: Optional[str]
    answer: str

# --- Simple heuristics ---
def _rule_route(q: str) -> Optional[Route]:
    ql = q.lower()
    if any(k in ql for k in ["weather", "temperature", "forecast", "rain", "humidity", "wind"]):
        return "weather"
    return None

def _maybe_city(q: str) -> Optional[str]:
    m = re.search(r"(?:in|at)\s+([A-Za-z][A-Za-z\s\-]{2,30})\??$", q.strip(), flags=re.I)
    return m.group(1).strip() if m else None

# --- Router node (rule-based with local LLM fallback) ---
def decide(state: State) -> State:
    q = state["question"]
    city = _maybe_city(q)
    route = _rule_route(q)

    if route is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify the user's query strictly as 'weather' or 'rag'. Reply with one word only."),
            ("human", "{q}")
        ])
        router = prompt | get_llm() | StrOutputParser()
        out = router.invoke({"q": q}).strip().lower()
        route = "weather" if "weather" in out else "rag"

    return {"route": route, "city": city}

# --- Tool nodes ---
def node_weather(state: State) -> State:
    city = state.get("city") or "Delhi"
    return {"answer": get_weather(city)}

def node_rag(state: State) -> State:
    return {"answer": answer_from_pdf(state["question"])}

# --- Build & compile graph ---
def build_graph():
    g = StateGraph(State)
    g.add_node("decide", decide)
    g.add_node("weather", node_weather)
    g.add_node("rag", node_rag)

    def _edge_router(s: State) -> Route:
        return s["route"]

    g.set_entry_point("decide")
    g.add_conditional_edges("decide", _edge_router, {"weather": "weather", "rag": "rag"})
    g.add_edge("weather", END)
    g.add_edge("rag", END)
    return g.compile()
