from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from backend.app.core.config import settings


class VectorService:

    def __init__(self, vector_size: int = 384) -> None:
        self.client = QdrantClient(
            url=settings.QDRANT_URL
        )

        self.collection_name = settings.QDRANT_COLLECTION

        self._ensure_collection(vector_size)

    def _ensure_collection(
        self,
        vector_size: int,
    ) -> None:

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

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            points.append(
                PointStruct(
                    id=index,
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
        limit: int = 5,
    ):

        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points
