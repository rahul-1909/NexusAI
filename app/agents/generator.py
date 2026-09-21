from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

llm = ChatGroq(
    model=settings.GROQ_MODEL,
    temperature=0.1,
    api_key=settings.GROQ_API_KEY
)

def rewrite_query(question: str) -> str:
    """Agent that reformulates a query to improve semantic vector search."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a search query optimizer. The initial retrieval failed to find good documents. "
                   "Rephrase the question into clear, technical search terms optimized for vector embeddings. "
                   "Return ONLY the reformulated query text."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    return chain.invoke({"question": question}).content.strip()

def generate_grounded_answer(question: str, context: str) -> str:
    """Agent that synthesizes a grounded answer using only the provided context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI Engineer. Answer the user's question using ONLY the facts provided in the Context. "
                   "Be concise, technical, and accurate. If the context does not contain the answer, "
                   "honestly say 'The provided knowledge base does not contain information to answer this question.'"),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context}).content.strip()

def generate_direct_answer(question: str) -> str:
    """Answers general questions or greetings directly."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and polite AI Engineering Assistant."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    return chain.invoke({"question": question}).content.strip()
