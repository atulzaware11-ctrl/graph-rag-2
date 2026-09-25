from app.services.document_service import load_pdf
from app.services.chunk_service import chunk_text
from app.services.graph_extraction_service import (
    GraphExtractionService,
)
from app.services.neo4j_service import Neo4jService


def ingest_graph(
    file_path: str,
    document_name: str,
    document_id: str,
) -> dict:

    pages = load_pdf(file_path)

    chunks = chunk_text(
        pages,
        document_id=document_id,
        filename=document_name,
    )

    extractor = GraphExtractionService()

    neo4j = Neo4jService()

    total_entities = 0
    total_relationships = 0

    try:

        for chunk in chunks:

            result = extractor.extract(
                text=chunk["text"],
                page=chunk["page"],
                chunk_id=chunk["chunk_id"],
            )

            entities = result.get(
                "entities",
                [],
            )

            relationships = result.get(
                "relationships",
                [],
            )

            neo4j.create_graph(
                document_name=document_name,
                document_id=document_id,
                page=chunk["page"],
                chunk_id=chunk["chunk_id"],
                entities=entities,
                relationships=relationships,
            )

            total_entities += len(entities)

            total_relationships += len(
                relationships
            )

    finally:

        neo4j.close()

    return {
        "chunks_processed": len(chunks),
        "entities_extracted": total_entities,
        "relationships_extracted": total_relationships,
    }
