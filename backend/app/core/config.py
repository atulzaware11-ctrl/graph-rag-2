import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv(
        "QDRANT_COLLECTION",
        "graphrag_documents",
    )

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    LLM_MODEL = os.getenv(
        "LLM_MODEL",
        "openrouter/free",
    )

    NEO4J_URI = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

    RERANKER_MODEL = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-macro-MiniLM-L-6-v2",
    )


settings = Settings()
