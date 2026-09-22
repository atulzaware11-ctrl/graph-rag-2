import re

from rank_bm25 import BM25Okapi


class BM25Service:

    def __init__(self):
        self.documents: list[dict] = []
        self.bm25 = None

    def build_index(
        self,
        documents: list[dict],
    ) -> None:

        self.documents = documents

        tokenized_documents = [
            self._tokenize(document["text"])
            for document in documents
        ]

        if tokenized_documents:
            self.bm25 = BM25Okapi(
                tokenized_documents
            )

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:

        if not self.bm25 or not self.documents:
            return []

        tokens = self._tokenize(query)

        scores = self.bm25.get_scores(tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indexes[:limit]:

            document = self.documents[index].copy()

            document["bm25_score"] = float(
                scores[index]
            )

            results.append(document)

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )