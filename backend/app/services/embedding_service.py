from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings


class EmbeddingService:

    def __init__(self) -> None:
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL
        )

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()
