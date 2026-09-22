import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from backend.app.pipelines.graph_ingestion import (ingest_graph)
from backend.app.pipelines.ingestion import ingest_pdf
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.llm_service import LLMService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="GraphRAG-X API",
    description="AI Research Intelligence Platform",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


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

    suffix = ".pdf"

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = temp_file.name

        result = ingest_pdf(temp_path)

        return {
            "message": "Document indexed successfully.",
            "filename": file.filename,
            **result,
        }

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/query")
async def query(request: QueryRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        retrieval = RetrievalService()

        contexts = retrieval.hybrid_search(
            request.question,
            request.top_k,
        )

        llm = LLMService()

        answer = llm.generate(
            request.question,
            contexts,
        )

        sources = [
            {
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
            }
            for item in contexts
        ]

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(contexts),
            "retrieval_method": "hybrid_vector_bm25",
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
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

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = temp_file.name

        result = ingest_graph(
            temp_path,
            file.filename,
        )

        return {
            "message": "Knowledge graph built successfully.",
            "filename": file.filename,
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