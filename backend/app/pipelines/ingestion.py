from backend.app.services.document_service import load_pdf
from backend.app.services.chunk_service import chunk_text
from backend.app.services.document_store import document_store
from backend.app.services.embedding_service import get_embedding_service
from backend.app.services.vector_service import get_vector_service
from backend.app.services.bm25_service import BM25Service


def ingest_pdf(
    file_path: str,
    document_id: str,
    filename: str,
    bm25_service: BM25Service | None = None,
) -> dict:
    pages = load_pdf(file_path)
    chunks = chunk_text(
        pages,
        document_id=document_id,
        filename=filename,
    )

    if not chunks:
        return {
            "document_id": document_id,
            "filename": filename,
            "pages_processed": len(pages),
            "chunks_processed": 0,
        }

    embeddings = get_embedding_service().embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    document_store.add_documents(chunks)
    get_vector_service().upsert(chunks, embeddings)
    if bm25_service is not None:
        bm25_service.build_index(document_id, chunks)

    return {
        "document_id": document_id,
        "filename": filename,
        "pages_processed": len(pages),
        "chunks_processed": len(chunks),
    }
