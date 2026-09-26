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
                   "Return ONLY the reformulated query text without quotes or explanations."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    return chain.invoke({"question": question}).content.strip()


def generate_grounded_answer(question: str, context: str) -> str:
    """Agent that synthesizes a grounded answer using provided context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI & Systems Engineer.\n\n"
                   "Follow these strict formatting guidelines:\n"
                   "- Provide a well-structured, easy-to-read technical explanation.\n"
                   "- Start with a clear 1-2 sentence executive overview.\n"
                   "- Use clean, bold bullet points for key concepts, architecture, and benefits.\n"
                   "- DO NOT generate ASCII or markdown tables (do not use '|---|---|' or pipe syntax) — use clean bullet lists instead.\n"
                   "- Separate distinct sections with clear double line breaks.\n"
                   "- If code is relevant, provide a concise, formatted python code snippet.\n"
                   "- Ensure output is crisp, highly readable, and professional."),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context}).content.strip()


def generate_direct_answer(question: str) -> str:
    """Answers technical or general questions directly using foundational domain knowledge."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI & Systems Engineer.\n\n"
                   "Follow these strict formatting guidelines:\n"
                   "- Provide a well-structured, easy-to-read technical explanation.\n"
                   "- Start with a clear 1-2 sentence executive overview.\n"
                   "- Use clean, bold bullet points for key concepts, architecture, and benefits.\n"
                   "- DO NOT generate ASCII or markdown tables (do not use '|---|---|' or pipe syntax) — use clean bullet lists instead.\n"
                   "- Separate distinct sections with clear double line breaks.\n"
                   "- If code is relevant, provide a concise, formatted python code snippet.\n"
                   "- Ensure output is crisp, highly readable, and professional."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    return chain.invoke({"question": question}).content.strip()
