from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

class RouteDecision(BaseModel):
    """Routing decision model."""
    destination: str = Field(
        description="Choose 'vectorstore' for technical/domain questions, or 'direct' for general chat/greetings."
    )

def route_question(question: str) -> str:
    """Agent that classifies the query destination."""
    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY
    )
    
    system_prompt = (
        "You are an expert query router. Classify the user question into either:\n"
        "- 'vectorstore': For questions about LangGraph, Qdrant, RAG, AI engineering, or technical concepts.\n"
        "- 'direct': For greetings, pleasantries, or general common knowledge.\n\n"
        "Return ONLY the word 'vectorstore' or 'direct'."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": question}).content.strip().lower()
    
    if "direct" in response:
        return "direct"
    return "vectorstore"
