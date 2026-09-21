from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Autonomous Multi-Agent RAG Engine"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Groq LLM (Free & Ultra-Fast)
    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    
    # FastEmbed Local Dense Embeddings (Free, runs on CPU)
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    
    # Qdrant Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "knowledge_base"
    QDRANT_API_KEY: str | None = None
    
    # FastAPI Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
