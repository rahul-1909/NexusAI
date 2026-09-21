from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

llm = ChatGroq(
    model=settings.GROQ_MODEL,
    temperature=0.2,
    api_key=settings.GROQ_API_KEY
)


def rewrite_query(question: str) -> str:
    """Agent that reformulates a query to improve semantic vector search."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a search query optimizer. Rephrase the question into clear, technical search terms optimized for vector embeddings. "
                   "Return ONLY the reformulated query text."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    return chain.invoke({"question": question}).content.strip()


def generate_grounded_answer(question: str, context: str) -> str:
    """Agent that synthesizes a grounded answer using provided context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI & Systems Engineer. Answer the user's question clearly, technically, and comprehensively using the provided Context. "
                   "Highlight core architectures, trade-offs, and operational benefits. Never refuse or claim you lack information; provide an insightful, authoritative technical answer."),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context}).content.strip()


def generate_direct_answer(question: str) -> str:
    """Answers technical or general questions directly using foundational domain knowledge."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI & Distributed Systems Engineer. Provide a clear, insightful, and comprehensive technical answer to the user's question."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    return chain.invoke({"question": question}).content.strip()
