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
        self.indexes: dict[str, tuple[list[dict], BM25Okapi | None]] = {}

    def build_index(self, document_id: str, documents: list[dict]) -> None:
        """
        Build the BM25 index from document chunks.
        """

        scoped_documents = [
            document.copy()
            for document in documents
            if document.get("document_id") == document_id
        ]

        tokenized_documents = [
            self._tokenize(
                document.get("text", "")
            )
            for document in scoped_documents
        ]

        if not tokenized_documents:
            self.indexes[document_id] = (scoped_documents, None)
            return

        self.indexes[document_id] = (
            scoped_documents,
            BM25Okapi(tokenized_documents),
        )

    def has_index(self, document_id: str) -> bool:
        return document_id in self.indexes

    def search(
        self,
        query: str,
        document_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Search indexed documents using BM25.
        """

        documents, bm25 = self.indexes.get(document_id, ([], None))
        if bm25 is None:
            return []

        if not documents:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = bm25.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[dict] = []

        for index in ranked_indexes[:limit]:
            document = documents[index].copy()

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
