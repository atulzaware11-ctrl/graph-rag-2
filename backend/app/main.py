import os
import shutil
import tempfile
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.pipelines.graph_ingestion import ingest_graph
from backend.app.pipelines.ingestion import ingest_pdf
from backend.app.services.reranker_service import RerankerService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.llm_service import LLMService
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.neo4j_service import Neo4jService


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
    question: str
    top_k: int = 5


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

    temp_path = None

    try:
        # ----------------------------------------------------
        # Save uploaded PDF temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = temp_file.name

        # ----------------------------------------------------
        # Ingest document
        # ----------------------------------------------------

        result = ingest_pdf(temp_path)

        return {
            "message": "Document indexed successfully.",
            "filename": file.filename,
            **result,
        }

    finally:
        # ----------------------------------------------------
        # Remove temporary file
        # ----------------------------------------------------

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


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

    try:

        # ====================================================
        # STEP 1 — RETRIEVAL
        # ====================================================

        retrieval_start = time.perf_counter()

        retrieval = RetrievalService()

        retrieved = retrieval.graph_enhanced_search(
            query=request.question,
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

        document_candidates = retrieved[
            "documents"
        ]

        graph_results = retrieved[
            "graph"
        ]

        # ====================================================
        # STEP 2 — RERANKING
        # ====================================================

        rerank_start = time.perf_counter()

        reranker = RerankerService()

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

        answer = llm.generate(
            question=request.question,
            contexts=llm_contexts,
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

        return {
            "question": request.question,

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

            "graph_context":
                graph_results,

            "timings":
                timings,
        }

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
    file: UploadFile = File(...)
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

    temp_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded PDF temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = temp_file.name

        # ----------------------------------------------------
        # Build Neo4j graph
        # ----------------------------------------------------

        result = ingest_graph(
            temp_path,
            file.filename,
        )

        return {
            "message":
                "Knowledge graph built successfully.",

            "filename":
                file.filename,

            **result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

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
    limit: int = 10,
):

    neo4j = Neo4jService()

    try:

        results = neo4j.search_graph(
            query,
            limit,
        )

        return {
            "query": query,
            "results": results,
            "result_count": len(results),
        }

    finally:

        neo4j.close()