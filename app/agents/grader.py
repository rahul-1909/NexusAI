from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings

llm = ChatGroq(
    model=settings.GROQ_MODEL,
    temperature=0,
    api_key=settings.GROQ_API_KEY
)

def grade_document_relevance(question: str, document_text: str) -> bool:
    """Agent that evaluates if a document chunk is relevant to the question."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question. "
                   "If the document contains keywords or semantic info related to the question, answer 'yes'. "
                   "Otherwise answer 'no'. Output ONLY 'yes' or 'no'."),
        ("human", "Question: {question}\n\nRetrieved Document:\n{document}")
    ])
    chain = prompt | llm
    res = chain.invoke({"question": question, "document": document_text}).content.strip().lower()
    return "yes" in res

def grade_hallucination_and_fidelity(context: str, generation: str) -> bool:
    """Agent that verifies the generated response is strictly grounded in the reference text."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator assessing whether an LLM answer is strictly grounded in and supported by "
                   "the provided facts. If the answer is fully supported, answer 'yes'. "
                   "If the answer invents facts not found in the context, answer 'no'. Output ONLY 'yes' or 'no'."),
        ("human", "Reference Facts:\n{context}\n\nGenerated Answer:\n{generation}")
    ])
    chain = prompt | llm
    res = chain.invoke({"context": context, "generation": generation}).content.strip().lower()
    return "yes" in res
