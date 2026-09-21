from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., example="How does an autonomous agent prevent LLM hallucinations?")

class SourceDoc(BaseModel):
    content: str
    score: float
    metadata: dict

class QueryResponse(BaseModel):
    query: str
    answer: str
    route: str
    is_grounded: bool
    sources: list[SourceDoc]
    latency_seconds: float
    execution_trace: list[str]

class IngestTextRequest(BaseModel):
    text: str = Field(..., example="FastAPI is a modern, fast web framework for building APIs with Python.")
    source_name: str = Field(default="user_input")

class IngestResponse(BaseModel):
    chunks_ingested: int
    source: str
    status: str

class BenchmarkRequest(BaseModel):
    queries: list[str] = Field(
        default=[
            "What is LangGraph and how does it manage state?",
            "Explain Qdrant vector indexing and cosine distance.",
            "How does an autonomous agent prevent LLM hallucinations?"
        ]
    )

class BenchmarkMetric(BaseModel):
    query: str
    latency_seconds: float
    route: str
    is_grounded: bool
    num_sources: int

class BenchmarkSummary(BaseModel):
    total_queries: int
    avg_latency_seconds: float
    p95_latency_seconds: float
    fidelity_rate: float
    metrics: list[BenchmarkMetric]
