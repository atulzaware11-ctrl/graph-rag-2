from backend.app.services.document_service import load_pdf
from backend.app.services.chunk_service import chunk_text
from backend.app.services.graph_extraction_service import (
    GraphExtractionService,
)
from backend.app.services.neo4j_service import Neo4jService


def ingest_graph(
    file_path: str,
    document_name: str,
) -> dict:

    pages = load_pdf(file_path)

    chunks = chunk_text(pages)

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