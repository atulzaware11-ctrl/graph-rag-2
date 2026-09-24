import os
from dotenv import load_dotenv

load_dotenv()
class Settings:
    OPENROUTER_API_KEY     = os.getenv("OPENROUTER_API_KEY","")
    QDRANT_URL : str       = os.getenv( " QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION :str = os.getenv(" QDRANT_COLLECTION","graphrag_documents")
    EMBEDDING_MODEL: str   = os.getenv("EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL : str        = os.getenv("LLM_MODEL","openrouter/free")
    NEO4J_URI : str        = os.getenv( "NEO4J_URI","")
    NEO4J_USERNAME:str     = os.getenv("NEO4J_USERNAME","neo4j")
    NEO4J_PASSWORD : str   = os.getenv("NEO4J_PASSWORD","")
    RERANKER_MODEL :str= os.getenv( "RERANKER_MODEL", "cross-encoder/ms-macro-MiniLM-L-6-v2",)
settings = Settings()