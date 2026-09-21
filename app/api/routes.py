import time
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.api.schemas import (
    QueryRequest, QueryResponse, SourceDoc,
    IngestTextRequest, IngestResponse,
    BenchmarkRequest, BenchmarkSummary, BenchmarkMetric
)
from app.ingestion.pipeline import ingestion_pipeline
from app.agents.graph import rag_workflow

router = APIRouter()

@router.post("/query", response_model=QueryResponse, summary="Query Multi-Agent RAG")
def query_endpoint(req: QueryRequest):
    """Executes the autonomous Multi-Agent RAG workflow."""
    start_time = time.perf_counter()
    
    initial_state = {
        "question": req.query,
        "documents": [],
        "rewritten_query": "",
        "generation": "",
        "route": "",
        "retry_count": 0,
        "is_grounded": False,
        "trace": []
    }
    
    try:
        final_state = rag_workflow.invoke(initial_state)
        latency = round(time.perf_counter() - start_time, 4)
        
        sources = [
            SourceDoc(
                content=d["content"],
                score=round(d.get("score", 0.0), 4),
                metadata=d.get("metadata", {})
            )
            for d in final_state.get("documents", [])
        ]
        
        return QueryResponse(
            query=req.query,
            answer=final_state.get("generation", "No response generated."),
            route=final_state.get("route", "unknown"),
            is_grounded=final_state.get("is_grounded", False),
            sources=sources,
            latency_seconds=latency,
            execution_trace=final_state.get("trace", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/text", response_model=IngestResponse, summary="Ingest Raw Text")
def ingest_text_endpoint(req: IngestTextRequest):
    """Ingests raw text directly into Qdrant collection."""
    try:
        result = ingestion_pipeline.process_and_ingest(req.text, req.source_name)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@router.post("/ingest/file", response_model=IngestResponse, summary="Upload File for Ingestion")
async def ingest_file_endpoint(file: UploadFile = File(...)):
    """Uploads and ingests a text or markdown file."""
    try:
        content = (await file.read()).decode("utf-8")
        result = ingestion_pipeline.process_and_ingest(content, source=file.filename)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {e}")

@router.post("/benchmark", response_model=BenchmarkSummary, summary="Run Automated RAG Benchmark")
def run_benchmark_endpoint(req: BenchmarkRequest):
    """Automated benchmark evaluating latency (P50/P95) and groundedness."""
    metrics: list[BenchmarkMetric] = []
    latencies = []
    grounded_count = 0
    
    for q in req.queries:
        start_time = time.perf_counter()
        state = {
            "question": q,
            "documents": [],
            "rewritten_query": "",
            "generation": "",
            "route": "",
            "retry_count": 0,
            "is_grounded": False,
            "trace": []
        }
        res = rag_workflow.invoke(state)
        lat = round(time.perf_counter() - start_time, 4)
        latencies.append(lat)
        
        is_grounded = res.get("is_grounded", False)
        if is_grounded:
            grounded_count += 1
            
        metrics.append(BenchmarkMetric(
            query=q,
            latency_seconds=lat,
            route=res.get("route", "unknown"),
            is_grounded=is_grounded,
            num_sources=len(res.get("documents", []))
        ))
        
    avg_lat = round(float(np.mean(latencies)), 4) if latencies else 0.0
    p95_lat = round(float(np.percentile(latencies, 95)), 4) if latencies else 0.0
    fidelity_rate = round(grounded_count / len(req.queries), 2) if req.queries else 0.0
    
    return BenchmarkSummary(
        total_queries=len(req.queries),
        avg_latency_seconds=avg_lat,
        p95_latency_seconds=p95_lat,
        fidelity_rate=fidelity_rate,
        metrics=metrics
    )
