from backend.app.services.embedding_service import (
    EmbeddingService,
)

from backend.app.services.vector_service import (
    VectorService,
)

from backend.app.services.bm25_service import (
    BM25Service,
)

from backend.app.services.document_store import (
    document_store,
)

from backend.app.services.neo4j_service import (
    Neo4jService,
)


class RetrievalService:
    """
    Central retrieval service for GraphRAG-X.

    Supported retrieval strategies:

    1. Vector search
    2. BM25 search
    3. Hybrid Vector + BM25 search
    4. Graph search
    5. Graph-enhanced retrieval
    """

    def __init__(self) -> None:
        # --------------------------------------------------
        # Initialize retrieval components
        # --------------------------------------------------

        self.embedding_service = (
            EmbeddingService()
        )

        self.vector_service = (
            VectorService()
        )

        self.bm25_service = (
            BM25Service()
        )

        self.neo4j_service = (
            Neo4jService()
        )

        # --------------------------------------------------
        # Load documents for BM25
        # --------------------------------------------------

        documents = (
            document_store.get_documents()
        )

        # The in-memory document store is empty
        # after a fresh Python process starts.
        #
        # Therefore, load the indexed chunks
        # directly from Qdrant when necessary.

        if not documents:
            documents = (
                self._load_documents_from_qdrant()
            )

        if documents:
            self.bm25_service.build_index(
                documents
            )

            print(
                f"BM25 index built with "
                f"{len(documents)} documents."
            )

        else:
            print(
                "WARNING: No documents available "
                "for BM25 indexing."
            )

    # ======================================================
    # QDRANT → BM25 DOCUMENT LOADING
    # ======================================================

    def _load_documents_from_qdrant(
        self,
    ) -> list[dict]:
        """
        Load document chunks from Qdrant.

        This allows BM25 to work even when the
        application starts in a fresh Python process
        and the in-memory DocumentStore is empty.
        """

        try:
            response = (
                self.vector_service.client.scroll(
                    collection_name=(
                        self.vector_service.collection_name
                    ),
                    limit=1000,
                    with_payload=True,
                )
            )

            points = response[0]

            documents: list[dict] = []

            for point in points:
                payload = (
                    point.payload or {}
                )

                chunk_id = payload.get(
                    "chunk_id"
                )

                page = payload.get(
                    "page"
                )

                text = payload.get(
                    "text"
                )

                # Ignore incomplete payloads.
                if not chunk_id:
                    continue

                if not text:
                    continue

                documents.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page,
                        "text": text,
                    }
                )

            # Keep deterministic ordering.
            documents.sort(
                key=lambda item: item.get(
                    "chunk_id",
                    "",
                )
            )

            print(
                f"Loaded {len(documents)} "
                f"chunks from Qdrant for BM25."
            )

            return documents

        except Exception as exc:
            print(
                "Failed to load documents "
                f"from Qdrant for BM25: {exc}"
            )

            return []

    # ======================================================
    # VECTOR SEARCH
    # ======================================================

    def vector_search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Semantic vector search using embeddings + Qdrant.
        """

        query_vector = (
            self.embedding_service.embed_query(
                query
            )
        )

        results = (
            self.vector_service.search(
                query_vector,
                limit=limit,
            )
        )

        documents: list[dict] = []

        for result in results:
            payload = (
                result.payload or {}
            )

            documents.append(
                {
                    "chunk_id": payload.get(
                        "chunk_id"
                    ),
                    "page": payload.get(
                        "page"
                    ),
                    "text": payload.get(
                        "text"
                    ),
                    "vector_score": float(
                        result.score
                    ),
                }
            )

        return documents

    # ======================================================
    # BM25 SEARCH
    # ======================================================

    def bm25_search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Keyword-based BM25 search.
        """

        return self.bm25_service.search(
            query,
            limit=limit,
        )

    # ======================================================
    # HYBRID SEARCH
    # ======================================================

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Combine semantic vector retrieval
        and BM25 keyword retrieval.

        Vector weight:
            0.6

        BM25 weight:
            0.4

        Reciprocal-rank style scoring is used
        to combine both result lists.
        """

        vector_results = (
            self.vector_search(
                query,
                limit=limit,
            )
        )

        bm25_results = (
            self.bm25_search(
                query,
                limit=limit,
            )
        )

        combined: dict[str, dict] = {}

        # --------------------------------------------------
        # Add vector results
        # --------------------------------------------------

        for rank, item in enumerate(
            vector_results,
            start=1,
        ):
            chunk_id = item.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            combined.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "page": item.get(
                        "page"
                    ),
                    "text": item.get(
                        "text"
                    ),
                    "score": 0.0,
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                },
            )

            combined[chunk_id][
                "score"
            ] += 0.6 / rank

            combined[chunk_id][
                "vector_score"
            ] = item.get(
                "vector_score",
                0.0,
            )

        # --------------------------------------------------
        # Add BM25 results
        # --------------------------------------------------

        for rank, item in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = item.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            combined.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "page": item.get(
                        "page"
                    ),
                    "text": item.get(
                        "text"
                    ),
                    "score": 0.0,
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                },
            )

            combined[chunk_id][
                "score"
            ] += 0.4 / rank

            combined[chunk_id][
                "bm25_score"
            ] = item.get(
                "bm25_score",
                0.0,
            )

        # --------------------------------------------------
        # Final ranking
        # --------------------------------------------------

        results = sorted(
            combined.values(),
            key=lambda item: item[
                "score"
            ],
            reverse=True,
        )

        return results[:limit]

    # ======================================================
    # GRAPH SEARCH
    # ======================================================

    def graph_search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search the Neo4j knowledge graph.
        """

        return (
            self.neo4j_service.search_graph(
                query,
                limit,
            )
        )

    # ======================================================
    # GRAPH-ENHANCED SEARCH
    # ======================================================

    def graph_enhanced_search(
        self,
        query: str,
        limit: int = 8,
        graph_limit: int = 10,
    ) -> dict:
        """
        Combine hybrid document retrieval
        with knowledge-graph retrieval.
        """

        hybrid_results = (
            self.hybrid_search(
                query,
                limit=limit,
            )
        )

        graph_results = (
            self.graph_search(
                query,
                limit=graph_limit,
            )
        )

        return {
            "documents": hybrid_results,
            "graph": graph_results,
        }

    # ======================================================
    # CLOSE CONNECTIONS
    # ======================================================

    def close(self) -> None:
        """
        Close Neo4j connection.
        """

        self.neo4j_service.close()