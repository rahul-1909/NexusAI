import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.config import settings
from app.api.routes import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous Multi-Agent RAG Engine utilizing LangGraph, Qdrant Vector Store, and FastAPI."
)

# --- SECURITY HEADERS MIDDLEWARE (Recommendations 1, 2, 3, 4) ---
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # 1. Content-Security-Policy (XSS Protection)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    
    # 2. Referrer-Policy and Permissions-Policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    
    # 3. Clickjacking Protection
    response.headers["X-Frame-Options"] = "DENY"
    
    # 4. MIME-Sniffing Protection
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    return response

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_V1_STR)

# Static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
def auto_seed_knowledge_base():
    """Automatically loads default knowledge base into Qdrant if collection is empty."""
    try:
        from app.db.qdrant_client import qdrant_service
        from app.ingestion.pipeline import ingestion_pipeline

        info = qdrant_service.get_collection_info()
        points_count = info.get("points_count", 0)

        if points_count == 0:
            doc_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs", "knowledge_base.txt")
            if os.path.exists(doc_path):
                with open(doc_path, "r", encoding="utf-8") as f:
                    content = f.read()
                ingestion_pipeline.process_and_ingest(content, source_name="knowledge_base.txt")
                print("[Startup] Successfully ingested knowledge_base.txt into Qdrant.")
    except Exception as e:
        print(f"[Startup Notice] Auto-ingest skipped: {e}")

@app.get("/", tags=["Frontend"])
def serve_frontend():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "NexusAI API is running. Visit /docs for Swagger UI."}

@app.get("/health", tags=["Health Check"])
def health_check():
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)