from uuid import uuid4
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from backend.app.core.config import settings


class VectorService:
    def __init__(self, vector_size: int = 384) -> None:
        # self.client = QdrantClient(url=settings.QDRANT_URL)
        self.client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY or None,
)
        self.collection_name = settings.QDRANT_COLLECTION

        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections()

        existing = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        points = []

        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=chunk,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        document_id: str,
        limit: int = 5,
    ):
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points

    def get_document_chunks(
        self,
        document_id: str,
        limit: int = 1000,
    ) -> list[dict]:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id,
                    )
                )
            ]
        )

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            record.payload
            for record in records
            if record.payload
        ]


@lru_cache(maxsize=1)
def get_vector_service() -> VectorService:
    return VectorService()
