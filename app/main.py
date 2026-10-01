import os
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.api.routes import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="NexusAI: Autonomous Multi-Agent Intelligence Copilot utilizing LangGraph, Qdrant, and FastAPI."
)

# --- 2. SPEED OPTIMIZATION: GZIP COMPRESSION (Reduces payload sizes by ~80%) ---
app.add_middleware(GZipMiddleware, minimum_size=500)

# --- 1. HTTP SECURITY HEADERS MIDDLEWARE (CSP, Clickjacking, MIME) ---


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)

    # 1. Content Security Policy (XSS Protection)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    # Security & Referrer Policies
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["X-Frame-Options"] = "DENY"
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
            doc_path = os.path.join(os.path.dirname(
                __file__), "..", "data", "sample_docs", "knowledge_base.txt")
            if os.path.exists(doc_path):
                with open(doc_path, "r", encoding="utf-8") as f:
                    content = f.read()
                ingestion_pipeline.process_and_ingest(
                    content, source_name="knowledge_base.txt")
                print(
                    "[Startup] Successfully ingested knowledge_base.txt into Qdrant.")
    except Exception as e:
        print(f"[Startup Notice] Auto-ingest skipped: {e}")

# --- 4. SEO EXTRAS: ROBOTS.TXT ---


@app.get("/robots.txt", response_class=PlainTextResponse, tags=["SEO"])
def robots_txt():
    return (
        "User-agent: *\n"
        "Allow: /\n"
        "Sitemap: https://multi-agent-autonomous-rag-engine.onrender.com/sitemap.xml\n"
    )

# --- 4. SEO EXTRAS: SITEMAP.XML ---


@app.get("/sitemap.xml", tags=["SEO"])
def sitemap_xml():
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://multi-agent-autonomous-rag-engine.onrender.com/</loc>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://multi-agent-autonomous-rag-engine.onrender.com/docs</loc>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
</urlset>"""
    return Response(content=xml_content, media_type="application/xml")

# --- FRONTEND ROUTE WITH CSP HEADERS ---


@app.get("/", tags=["Frontend"])
def serve_frontend():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        response = FileResponse(index_file)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "font-src 'self' https://cdnjs.cloudflare.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response
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
    uvicorn.run("app.main:app", host=settings.API_HOST,
                port=settings.API_PORT, reload=True)
