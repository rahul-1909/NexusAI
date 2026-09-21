import logging
from langgraph.graph import END, StateGraph
from app.agents.state import AgentState
from app.agents.router import route_question
from app.agents.grader import grade_document_relevance, grade_hallucination_and_fidelity
from app.agents.generator import rewrite_query, generate_grounded_answer, generate_direct_answer
from app.ingestion.pipeline import ingestion_pipeline
from app.db.qdrant_client import qdrant_service

logger = logging.getLogger(__name__)

# --- NODE DEFINITIONS ---


def router_node(state: AgentState) -> dict:
    question = state["question"]
    route = route_question(question)
    trace = state.get("trace", [])
    trace.append(f"Router Agent evaluated question -> Route: '{route}'")
    return {"route": route, "trace": trace, "retry_count": state.get("retry_count", 0)}


def retriever_node(state: AgentState) -> dict:
    search_query = state.get("rewritten_query") or state["question"]
    query_vector = ingestion_pipeline.embed_text(search_query)
    docs = qdrant_service.search_vectors(query_vector=query_vector, top_k=3)
    trace = state.get("trace", [])
    trace.append(f"Retriever Agent fetched {len(docs)} chunks from Qdrant")
    return {"documents": docs, "trace": trace}


def grade_documents_node(state: AgentState) -> dict:
    question = state["question"]
    documents = state.get("documents", [])
    relevant_docs = []
    trace = state.get("trace", [])

    for doc in documents:
        is_rel = grade_document_relevance(question, doc["content"])
        if is_rel:
            relevant_docs.append(doc)

    trace.append(
        f"Grader Agent validated {len(relevant_docs)} / {len(documents)} chunks as relevant")
    return {"documents": relevant_docs, "trace": trace}


def rewriter_node(state: AgentState) -> dict:
    question = state["question"]
    rewritten = rewrite_query(question)
    retry_count = state.get("retry_count", 0) + 1
    trace = state.get("trace", [])
    trace.append(
        f"Rewriter Agent reformulated query (Attempt {retry_count}): '{rewritten}'")
    return {"rewritten_query": rewritten, "retry_count": retry_count, "trace": trace}


def generator_node(state: AgentState) -> dict:
    question = state["question"]
    docs = state.get("documents", [])
    trace = state.get("trace", [])

    if docs:
        context = "\n\n".join(
            [f"[Source: {d.get('metadata', {}).get('source', 'unknown')}]\n{d['content']}" for d in docs])
        generation = generate_grounded_answer(question, context)
        trace.append(
            "Generator Agent synthesized response from verified context")
    else:
        generation = generate_direct_answer(question)
        trace.append(
            "Generator Agent synthesized response from foundational AI knowledge")

    return {"generation": generation, "trace": trace}


def direct_llm_node(state: AgentState) -> dict:
    question = state["question"]
    generation = generate_direct_answer(question)
    trace = state.get("trace", [])
    trace.append("Direct LLM Agent responded to general greeting")
    return {"generation": generation, "is_grounded": True, "trace": trace}


def hallucination_checker_node(state: AgentState) -> dict:
    docs = state.get("documents", [])
    generation = state["generation"]
    trace = state.get("trace", [])

    if not docs:
        trace.append(
            "Hallucination Grader verified foundational technical response")
        return {"is_grounded": True, "trace": trace}

    context = "\n\n".join([d["content"] for d in docs])
    is_grounded = grade_hallucination_and_fidelity(context, generation)
    trace.append(f"Hallucination Grader verified groundedness: {is_grounded}")
    return {"is_grounded": is_grounded, "trace": trace}

# --- CONDITIONAL ROUTING FUNCTIONS ---


def decide_route(state: AgentState) -> str:
    return state["route"]


def decide_to_generate(state: AgentState) -> str:
    # If no docs passed the grader and we haven't retried twice, rewrite!
    if not state.get("documents") and state.get("retry_count", 0) < 2:
        return "rewrite"
    return "generate"


def decide_fidelity(state: AgentState) -> str:
    if state.get("is_grounded", False) or state.get("retry_count", 0) >= 2:
        return "done"
    return "regenerate"

# --- WORKFLOW GRAPH COMPILATION ---


builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("retriever", retriever_node)
builder.add_node("grade_docs", grade_documents_node)
builder.add_node("rewriter", rewriter_node)
builder.add_node("generator", generator_node)
builder.add_node("direct_llm", direct_llm_node)
builder.add_node("hallucination_checker", hallucination_checker_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    decide_route,
    {
        "vectorstore": "retriever",
        "direct": "direct_llm"
    }
)

builder.add_edge("retriever", "grade_docs")

builder.add_conditional_edges(
    "grade_docs",
    decide_to_generate,
    {
        "rewrite": "rewriter",
        "generate": "generator"
    }
)

builder.add_edge("rewriter", "retriever")
builder.add_edge("generator", "hallucination_checker")

builder.add_conditional_edges(
    "hallucination_checker",
    decide_fidelity,
    {
        "done": END,
        "regenerate": "generator"
    }
)

builder.add_edge("direct_llm", END)

# Compiled executable graph
rag_workflow = builder.compile()
