import os
import logging
import tempfile
import time
from functools import lru_cache
from pathlib import PurePath
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError, PdfStreamError

from backend.app.pipelines.graph_ingestion import ingest_graph
from backend.app.pipelines.ingestion import ingest_pdf
from backend.app.services.reranker_service import RerankerService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.llm_service import LLMQuotaExceededError, LLMService
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.neo4j_service import Neo4jService
from backend.app.services.document_registry import document_registry
from backend.app.services.bm25_service import BM25Service
from backend.app.core.config import settings

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
LLM_QUOTA_MESSAGE = (
    "The AI generation service has reached its current free-model request limit. "
    "Retrieval and document indexing are still working. Please try again after the quota resets."
)
bm25_service = BM25Service()


@lru_cache(maxsize=1)
def _get_reranker() -> RerankerService:
    return RerankerService()


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="GraphRAG-X API",
    description="AI Research Intelligence Platform",
    version="0.1.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):
    document_id: UUID
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.CORS_ORIGINS.split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "healthy"
    }


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    original_name = file.filename or ""
    filename = PurePath(original_name.replace("\\", "/")).name
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )
    if file.content_type and file.content_type.lower() not in {
        "application/pdf", "application/octet-stream"
    }:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = None
    document_id = str(uuid4())
    document_registry.create_document(document_id, filename)

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_path = temp_file.name
            total_bytes = 0
            header = b""
            while content := await file.read(1024 * 1024):
                total_bytes += len(content)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB upload limit.")
                if len(header) < 5:
                    header += content[: 5 - len(header)]
                temp_file.write(content)
        if header != b"%PDF-":
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        document_registry.update_document(document_id, status="indexing")
        result = ingest_pdf(temp_path, document_id, filename, bm25_service)
        metadata = document_registry.update_document(
            document_id,
            status="ready",
            pages_processed=result["pages_processed"],
            chunks_processed=result["chunks_processed"],
        )

        return {
            **(metadata or {}),
            "status": "ready",
        }
    except HTTPException:
        document_registry.update_document(document_id, status="failed")
        raise
    except (PdfReadError, PdfStreamError) as exc:
        logger.info("Rejected malformed PDF for document %s", document_id)
        document_registry.update_document(document_id, status="failed")
        raise HTTPException(status_code=400, detail="The uploaded PDF is malformed or unreadable.") from exc
    except Exception as exc:
        logger.exception("Document indexing failed for %s", document_id)
        document_registry.update_document(document_id, status="failed")
        raise HTTPException(status_code=500, detail="Document indexing failed.") from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()


# ============================================================
# MAIN GRAPH RAG QUERY
# ============================================================

@app.post("/api/query")
def query(request: QueryRequest):

    # ========================================================
    # TOTAL TIMER
    # ========================================================

    total_start = time.perf_counter()

    retrieval = None
    document_id = str(request.document_id)

    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    metadata = document_registry.get_document(document_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if metadata.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Document is not ready for querying.")

    try:

        # ====================================================
        # STEP 1 — RETRIEVAL
        # ====================================================

        retrieval_start = time.perf_counter()

        retrieval = RetrievalService(bm25_service=bm25_service)

        retrieved = retrieval.route_search(
            query=request.question,
            document_id=document_id,
            limit=max(
                request.top_k * 3,
                10,
            ),
            graph_limit=10,
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        document_candidates = [
            candidate for candidate in retrieved["documents"]
            if candidate.get("document_id") == document_id
        ]

        graph_results = [
            item for item in retrieved["graph"]
            if item.get("document_id") == document_id
        ]

        # ====================================================
        # STEP 2 — RERANKING
        # ====================================================

        rerank_start = time.perf_counter()

        reranker = _get_reranker()

        reranked_documents = reranker.rerank(
            query=request.question,
            documents=document_candidates,
            top_k=request.top_k,
        )

        rerank_time = (
            time.perf_counter()
            - rerank_start
        )

        # ====================================================
        # STEP 3 — EVIDENCE PROCESSING
        # ====================================================

        evidence_start = time.perf_counter()

        evidence_service = EvidenceService()

        sources = evidence_service.build_sources(
            reranked_documents
        )

        evidence_time = (
            time.perf_counter()
            - evidence_start
        )

        # ====================================================
        # STEP 4 — BUILD GRAPH CONTEXT
        # ====================================================

        graph_start = time.perf_counter()

        graph_context = []

        for graph_item in graph_results:

            entity = graph_item.get(
                "entity",
                "",
            )

            entity_type = graph_item.get(
                "entity_type",
                "",
            )

            if not entity:
                continue

            # Add entity information
            graph_context.append(
                f"Entity: {entity}\n"
                f"Type: {entity_type}"
            )

            # Add relationships
            for relationship in graph_item.get(
                "relationships",
                [],
            ):

                relation = relationship.get(
                    "relationship"
                )

                connected_entity = relationship.get(
                    "connected_entity"
                )

                connected_type = relationship.get(
                    "connected_type"
                )

                if not connected_entity:
                    continue

                graph_context.append(
                    f"{entity} "
                    f"-[{relation}]-> "
                    f"{connected_entity} "
                    f"({connected_type})"
                )

        graph_time = (
            time.perf_counter()
            - graph_start
        )

        # ====================================================
        # STEP 5 — BUILD LLM CONTEXT
        # ====================================================

        context_start = time.perf_counter()

        llm_contexts = []

        for source in sources:

            llm_contexts.append(
                {
                    "document_id": source["document_id"],
                    "chunk_id": source[
                        "chunk_id"
                    ],
                    "page": source[
                        "page"
                    ],
                    "text": source[
                        "text"
                    ],
                    "rerank_score": source[
                        "rerank_score"
                    ],
                }
            )

        # ----------------------------------------------------
        # Add knowledge graph separately
        # ----------------------------------------------------

        if graph_context:

            graph_text = (
                "KNOWLEDGE GRAPH EVIDENCE:\n\n"
                + "\n".join(
                    graph_context
                )
            )

            llm_contexts.append(
                {
                    "document_id": document_id,
                    "chunk_id": "knowledge_graph",
                    "page": None,
                    "text": graph_text,
                    "rerank_score": None,
                }
            )

        context_time = (
            time.perf_counter()
            - context_start
        )

        # ====================================================
        # STEP 6 — LLM GENERATION
        # ====================================================

        llm_start = time.perf_counter()

        llm = LLMService()
        llm_quota_exceeded = False
        try:
            answer = llm.generate(
                question=request.question,
                contexts=llm_contexts,
            )
        except LLMQuotaExceededError:
            logger.warning("LLM generation quota reached for document %s", document_id)
            llm_quota_exceeded = True
            answer = (
                "AI answer generation is temporarily unavailable. "
                "Retrieved evidence is available below."
            )

        llm_time = (
            time.perf_counter()
            - llm_start
        )

        # ====================================================
        # STEP 7 — CONFIDENCE
        # ====================================================

        confidence_start = time.perf_counter()

        confidence = (
            evidence_service.calculate_confidence(
                sources=sources,
                graph_entities=len(
                    graph_results
                ),
            )
        )

        confidence_time = (
            time.perf_counter()
            - confidence_start
        )

        # ====================================================
        # STEP 8 — CITATION VALIDATION
        # ====================================================

        citation_start = time.perf_counter()

        citation_validation = (
            evidence_service.validate_citations(
                answer=answer,
                sources=sources,
            )
        )

        citation_time = (
            time.perf_counter()
            - citation_start
        )

        # ====================================================
        # TOTAL TIME
        # ====================================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        # ====================================================
        # PERFORMANCE METRICS
        # ====================================================

        timings = {
            "retrieval_seconds": round(
                retrieval_time,
                4,
            ),

            "rerank_seconds": round(
                rerank_time,
                4,
            ),

            "evidence_seconds": round(
                evidence_time,
                4,
            ),

            "graph_seconds": round(
                graph_time,
                4,
            ),

            "context_build_seconds": round(
                context_time,
                4,
            ),

            "llm_seconds": round(
                llm_time,
                4,
            ),

            "confidence_seconds": round(
                confidence_time,
                4,
            ),

            "citation_seconds": round(
                citation_time,
                4,
            ),

            "total_seconds": round(
                total_time,
                4,
            ),
        }

        # ====================================================
        # PRINT PERFORMANCE TO TERMINAL
        # ====================================================

        print()
        print("=" * 65)
        print("                 GRAPHRAG-X PERFORMANCE")
        print("=" * 65)

        print(
            f"Retrieval       : "
            f"{timings['retrieval_seconds']}s"
        )

        print(
            f"Reranker        : "
            f"{timings['rerank_seconds']}s"
        )

        print(
            f"Evidence        : "
            f"{timings['evidence_seconds']}s"
        )

        print(
            f"Graph Context   : "
            f"{timings['graph_seconds']}s"
        )

        print(
            f"Context Build   : "
            f"{timings['context_build_seconds']}s"
        )

        print(
            f"LLM             : "
            f"{timings['llm_seconds']}s"
        )

        print(
            f"Confidence      : "
            f"{timings['confidence_seconds']}s"
        )

        print(
            f"Citation Check  : "
            f"{timings['citation_seconds']}s"
        )

        print("-" * 65)

        print(
            f"TOTAL           : "
            f"{timings['total_seconds']}s"
        )

        print("=" * 65)
        print()

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        response_payload = {
            "question": request.question,
            "document_id": document_id,

            "answer": answer,

            "confidence": confidence,

            "citation_validation":
                citation_validation,

            "sources": sources,

            "retrieved_chunks":
                len(document_candidates),

            "reranked_chunks":
                len(reranked_documents),

            "graph_entities":
                len(graph_results),

            "retrieval_method":
                "hybrid_vector_bm25_graph_reranked",

            "retrieval_route": retrieved["retrieval_route"],

            "route_reason": retrieved["route_reason"],

            "graph_context":
                graph_results,

            "timings":
                timings,
        }

        if llm_quota_exceeded:
            response_payload.update(
                {
                    "error": "llm_quota_exceeded",
                    "message": LLM_QUOTA_MESSAGE,
                    "retrieval_available": True,
                }
            )
            return JSONResponse(status_code=429, content=response_payload)

        return response_payload

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed for document %s", document_id)
        raise HTTPException(status_code=500, detail="Query failed.") from exc

    finally:

        # ====================================================
        # CLOSE NEO4J CONNECTION
        # ====================================================

        if retrieval:
            retrieval.close()


# ============================================================
# BUILD KNOWLEDGE GRAPH
# ============================================================

@app.post("/api/graph/build")
async def build_graph(
    document_id: UUID = Form(...),
    file: UploadFile = File(...),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    document_key = str(document_id)
    metadata = document_registry.get_document(document_key)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    supplied_name = PurePath(file.filename.replace("\\", "/")).name
    if supplied_name != metadata["filename"]:
        raise HTTPException(status_code=400, detail="Uploaded filename does not match the selected document.")
    if file.content_type and file.content_type.lower() not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded PDF temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_path = temp_file.name
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB upload limit.")
            temp_file.write(content)
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        # ----------------------------------------------------
        # Build Neo4j graph
        # ----------------------------------------------------

        result = ingest_graph(
            temp_path,
            metadata["filename"],
            document_key,
        )
        document_registry.update_document(document_key, graph_status="ready")

        return {
            "message":
                "Knowledge graph built successfully.",

            "filename": metadata["filename"],

            **result,
        }

    except HTTPException:
        raise
    except LLMQuotaExceededError:
        return JSONResponse(
            status_code=429,
            content={
                "error": "llm_quota_exceeded",
                "message": "Knowledge graph generation has reached the current free-model request limit.",
            },
        )
    except (PdfReadError, PdfStreamError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded PDF is malformed or unreadable.") from exc
    except Exception as exc:
        logger.exception("Graph build failed for document %s", document_key)
        raise HTTPException(status_code=500, detail="Knowledge graph build failed.") from exc

    finally:

        if temp_path and os.path.exists(
            temp_path
        ):
            os.remove(temp_path)


# ============================================================
# GRAPH SEARCH
# ============================================================

@app.get("/api/graph/search")
def graph_search(
    query: str,
    document_id: UUID,
    limit: int = 10,
):

    neo4j = Neo4jService()

    try:

        results = neo4j.search_graph(
            query,
            str(document_id),
            limit,
        )

        return {
            "query": query,
            "results": results,
            "result_count": len(results),
        }

    finally:

        neo4j.close()
