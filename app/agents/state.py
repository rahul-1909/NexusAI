from typing import TypedDict

class AgentState(TypedDict):
    question: str
    documents: list[dict]
    rewritten_query: str
    generation: str
    route: str          # 'vectorstore' or 'direct'
    retry_count: int    # Prevents infinite loops
    is_grounded: bool   # Anti-hallucination flag
    trace: list[str]    # Step-by-step audit log of what agents did
