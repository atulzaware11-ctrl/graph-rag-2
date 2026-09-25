import asyncio
from types import SimpleNamespace

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import backend.app.main as main_module
import backend.app.services.graph_extraction_service as graph_module
import backend.app.services.llm_service as llm_module
from backend.app.main import QueryRequest, app
from backend.app.services.llm_service import LLMQuotaExceededError


DOC_ID = "33333333-3333-4333-8333-333333333333"


class FakeRegistry:
    def __init__(self, metadata=None):
        self.metadata = metadata
        self.created = []
        self.updates = []

    def create_document(self, document_id, filename):
        self.created.append((document_id, filename))

    def get_document(self, document_id):
        if self.metadata and document_id == self.metadata["document_id"]:
            return self.metadata.copy()
        return None

    def update_document(self, document_id, **updates):
        self.updates.append((document_id, updates))
        if self.metadata and document_id == self.metadata["document_id"]:
            self.metadata.update(updates)
            return self.metadata.copy()
        return None


def document_registry_for(document_id=DOC_ID):
    return FakeRegistry(
        {
            "document_id": document_id,
            "filename": "research.pdf",
            "status": "ready",
        }
    )


def test_upload_does_not_call_graph_extraction(monkeypatch):
    registry = FakeRegistry()
    graph_calls = []

    def fail_graph(*args):
        graph_calls.append(args)
        raise AssertionError("upload must not build the graph")

    monkeypatch.setattr(main_module, "document_registry", registry)
    monkeypatch.setattr(main_module, "ingest_pdf", lambda *args: {
        "pages_processed": 1,
        "chunks_processed": 2,
    })
    monkeypatch.setattr(main_module, "ingest_graph", fail_graph)

    response = TestClient(app).post(
        "/api/documents/upload",
        files={"file": ("research.pdf", b"%PDF-test", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert graph_calls == []


def test_graph_build_calls_graph_extraction_with_document_id(monkeypatch):
    registry = document_registry_for()
    graph_calls = []

    def fake_ingest_graph(file_path, document_name, document_id):
        graph_calls.append((file_path, document_name, document_id))
        return {
            "chunks_processed": 2,
            "entities_extracted": 3,
            "relationships_extracted": 1,
        }

    monkeypatch.setattr(main_module, "document_registry", registry)
    monkeypatch.setattr(main_module, "ingest_graph", fake_ingest_graph)

    response = TestClient(app).post(
        "/api/graph/build",
        data={"document_id": DOC_ID},
        files={"file": ("research.pdf", b"%PDF-test", "application/pdf")},
    )

    assert response.status_code == 200
    assert graph_calls and graph_calls[0][1:] == ("research.pdf", DOC_ID)
    assert registry.updates[-1] == (DOC_ID, {"graph_status": "ready"})


def test_graph_build_quota_error_is_clean_429(monkeypatch):
    monkeypatch.setattr(main_module, "document_registry", document_registry_for())
    monkeypatch.setattr(
        main_module,
        "ingest_graph",
        lambda *args: (_ for _ in ()).throw(LLMQuotaExceededError("quota")),
    )

    response = TestClient(app).post(
        "/api/graph/build",
        data={"document_id": DOC_ID},
        files={"file": ("research.pdf", b"%PDF-test", "application/pdf")},
    )

    assert response.status_code == 429
    assert response.json() == {
        "error": "llm_quota_exceeded",
        "message": "Knowledge graph generation has reached the current free-model request limit.",
    }


def test_query_quota_error_is_clean_429(monkeypatch):
    monkeypatch.setattr(main_module, "document_registry", document_registry_for())

    class FakeRetrieval:
        def __init__(self, bm25_service=None):
            pass

        def graph_enhanced_search(self, **kwargs):
            return {"documents": [], "graph": []}

        def close(self):
            pass

    class FakeReranker:
        def rerank(self, **kwargs):
            return []

    monkeypatch.setattr(main_module, "RetrievalService", FakeRetrieval)
    monkeypatch.setattr(main_module, "_get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(
        main_module.LLMService,
        "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(LLMQuotaExceededError("quota")),
    )

    response = main_module.query(QueryRequest(document_id=DOC_ID, question="What is this?"))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 429
    assert b"llm_quota_exceeded" in response.body
    assert b"retrieval_available" in response.body


def test_openai_clients_disable_sdk_retries(monkeypatch):
    captured = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(llm_module.settings, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(graph_module.settings, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(graph_module, "OpenAI", FakeOpenAI)

    llm_module.LLMService()
    graph_module.GraphExtractionService()

    assert len(captured) == 2
    assert all(client["max_retries"] == 0 for client in captured)
    assert all(client["base_url"] == "https://openrouter.ai/api/v1" for client in captured)
