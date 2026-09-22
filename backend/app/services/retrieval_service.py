from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_service import VectorService
from backend.app.services.bm25_service import BM25Service
from backend.app.services.document_store import document_store


class RetrievalService:

    def __init__(self) -> None:

        self.embedding_service = EmbeddingService()

        self.vector_service = VectorService()

        self.bm25_service = BM25Service()

        documents = document_store.get_documents()

        if documents:
            self.bm25_service.build_index(
                documents
            )

    def vector_search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        query_vector = (
            self.embedding_service.embed_query(
                query
            )
        )

        results = self.vector_service.search(
            query_vector,
            limit=limit,
        )

        documents = []

        for result in results:

            payload = result.payload or {}

            documents.append(
                {
                    "chunk_id": payload.get(
                        "chunk_id"
                    ),
                    "page": payload.get("page"),
                    "text": payload.get("text"),
                    "vector_score": float(
                        result.score
                    ),
                }
            )

        return documents

    def bm25_search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        return self.bm25_service.search(
            query,
            limit=limit,
        )

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        vector_results = self.vector_search(
            query,
            limit,
        )

        bm25_results = self.bm25_search(
            query,
            limit,
        )

        combined = {}

        # Vector contribution
        for rank, item in enumerate(
            vector_results,
            start=1,
        ):

            chunk_id = item["chunk_id"]

            combined.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "page": item["page"],
                    "text": item["text"],
                    "score": 0.0,
                },
            )

            combined[chunk_id]["score"] += (
                0.6 / rank
            )

        # BM25 contribution
        for rank, item in enumerate(
            bm25_results,
            start=1,
        ):

            chunk_id = item["chunk_id"]

            combined.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "page": item["page"],
                    "text": item["text"],
                    "score": 0.0,
                },
            )

            combined[chunk_id]["score"] += (
                0.4 / rank
            )

        results = sorted(
            combined.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:limit]