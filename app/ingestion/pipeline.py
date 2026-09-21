import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from qdrant_client.http import models
from app.core.config import settings
from app.db.qdrant_client import qdrant_service

class IngestionPipeline:
    def __init__(self):
        self.embedding_model = TextEmbedding(settings.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " "]
        )

    def embed_text(self, text: str) -> list[float]:
        """Embeds a single string into a 384-dim vector."""
        return list(self.embedding_model.embed([text]))[0].tolist()

    def process_and_ingest(self, text: str, source: str = "manual_upload") -> dict:
        """Splits raw text, computes dense embeddings, and stores in Qdrant."""
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            return {"chunks_ingested": 0, "status": "empty text"}

        # Generate vectors in batch
        vectors = [v.tolist() for v in self.embedding_model.embed(chunks)]

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": chunk,
                        "metadata": {
                            "source": source,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    }
                )
            )

        qdrant_service.upsert_chunks(points)
        return {
            "chunks_ingested": len(points),
            "source": source,
            "status": "success"
        }

ingestion_pipeline = IngestionPipeline()
