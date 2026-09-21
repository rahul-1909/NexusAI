import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=1.5,
                check_compatibility=False
            )
            self.client.get_collections()
            logger.info("Connected to remote/Docker Qdrant.")
        except Exception:
            logger.info("Using local embedded disk storage at './qdrant_storage'.")
            self.client = QdrantClient(path="./qdrant_storage")

        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.EMBEDDING_DIMENSION
        self.ensure_collection_exists()

    def ensure_collection_exists(self):
        """Creates the collection if it does not already exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def search_vectors(self, query_vector: list[float], top_k: int = 4) -> list[dict]:
        """Performs dense vector similarity search using modern query_points."""
        # query_points works on both remote Docker and local disk storage
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        return [
            {
                "id": str(point.id),
                "score": round(float(point.score), 4),
                "content": point.payload.get("content", ""),
                "metadata": point.payload.get("metadata", {})
            }
            for point in response.points
        ]

    def upsert_chunks(self, points: list[models.PointStruct]):
        """Upserts embedded document points into Qdrant."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )

# Global singleton
qdrant_service = QdrantService()
