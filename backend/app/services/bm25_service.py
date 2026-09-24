import re

from rank_bm25 import BM25Okapi


class BM25Service:
    """
    BM25 keyword-based retrieval service.

    The service receives document chunks from the retrieval layer,
    builds a BM25 index, and returns the highest-scoring chunks
    for a query.
    """

    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def build_index(self, documents: list[dict]) -> None:
        """
        Build the BM25 index from document chunks.
        """

        self.documents = documents

        tokenized_documents = [
            self._tokenize(
                document.get("text", "")
            )
            for document in documents
        ]

        if not tokenized_documents:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Search indexed documents using BM25.
        """

        if self.bm25 is None:
            return []

        if not self.documents:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[dict] = []

        for index in ranked_indexes[:limit]:
            document = self.documents[index].copy()

            document["bm25_score"] = float(
                scores[index]
            )

            results.append(document)

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple lowercase alphanumeric tokenizer.
        """

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )