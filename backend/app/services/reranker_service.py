from sentence_transformers import CrossEncoder


class RerankerService:
    """
    Reranks retrieved document chunks using a CrossEncoder.

    Unlike normal embedding search, a CrossEncoder receives
    both the question and candidate document together.

    Example:

        Question:
        "Which model was evaluated?"

        Candidate:
        "The authors evaluated the proposed model..."

    The model then produces a relevance score.

    This is useful after initial retrieval because we can
    retrieve a larger candidate pool first and then select
    the most relevant evidence.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:

        # The model is downloaded the first time it is used.
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank retrieved documents according to their
        semantic relevance to the query.
        """

        if not documents:
            return []

        # Create question/document pairs.
        pairs = [
            [
                query,
                document.get("text", ""),
            ]
            for document in documents
        ]

        # Calculate relevance scores.
        scores = self.model.predict(pairs)

        reranked = []

        for document, score in zip(
            documents,
            scores,
        ):

            item = document.copy()

            item["rerank_score"] = float(score)

            reranked.append(item)

        # Highest relevance first.
        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]