from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.bm25_service import BM25Service
from backend.app.services.document_registry import DocumentRegistry
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.vector_service import VectorService
from backend.app.services.neo4j_service import Neo4jService


DOC_A = "11111111-1111-4111-8111-111111111111"
DOC_B = "22222222-2222-4222-8222-222222222222"


def test_bm25_indexes_are_isolated_by_document():
    service = BM25Service()
    service.build_index(DOC_A, [{"document_id": DOC_A, "chunk_id": "a", "text": "alpha only"}])
    service.build_index(DOC_B, [{"document_id": DOC_B, "chunk_id": "b", "text": "beta only"}])

    results = service.search("beta", DOC_A)
    assert all(item["document_id"] == DOC_A for item in results)
    assert not any("beta" in item["text"] for item in results)


def test_qdrant_search_always_applies_document_filter():
    captured = {}
    service = VectorService.__new__(VectorService)
    service.collection_name = "graphrag_documents_v2"
    service.client = SimpleNamespace(
        query_points=lambda **kwargs: (captured.update(kwargs) or SimpleNamespace(points=[]))
    )
    service.search([0.1, 0.2], document_id=DOC_A)
    assert captured["query_filter"].must[0].key == "document_id"
    assert captured["query_filter"].must[0].match.value == DOC_A


def test_qdrant_point_ids_are_uuid_and_unique():
    class Client:
        points = None

        def upsert(self, **kwargs):
            self.points = kwargs["points"]

    service = VectorService.__new__(VectorService)
    service.collection_name = "graphrag_documents_v2"
    service.client = Client()
    service.upsert(
        [
            {"document_id": DOC_A, "filename": "a.pdf", "chunk_id": "chunk_0", "page": 1, "text": "A"},
            {"document_id": DOC_B, "filename": "b.pdf", "chunk_id": "chunk_0", "page": 1, "text": "B"},
        ],
        [[0.1], [0.2]],
    )
    ids = [point.id for point in service.client.points]
    assert len(set(ids)) == 2
    assert all(isinstance(UUID(point_id), UUID) for point_id in ids)


def test_hybrid_results_keep_only_requested_document():
    service = RetrievalService.__new__(RetrievalService)
    service.vector_search = lambda query, document_id, limit: [
        {"document_id": document_id, "chunk_id": "a", "text": "safe", "page": 1}
    ]
    service.bm25_search = lambda query, document_id, limit: [
        {"document_id": document_id, "chunk_id": "a", "text": "safe", "page": 1}
    ]
    results = RetrievalService.hybrid_search(service, "query", DOC_A)
    assert results and all(item["document_id"] == DOC_A for item in results)


def test_evidence_preserves_document_ownership():
    sources = EvidenceService().build_sources([
        {"document_id": DOC_A, "filename": "a.pdf", "chunk_id": "a", "page": 1, "text": "evidence", "rerank_score": 1.0}
    ])
    assert sources[0]["document_id"] == DOC_A


def test_graph_query_is_scoped_to_document_id():
    captured = {}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def run(self, query, **params):
            captured["query"] = query
            captured["params"] = params
            return []

    service = Neo4jService.__new__(Neo4jService)
    service.driver = SimpleNamespace(session=lambda: Session())
    assert service.search_graph("methodology", DOC_A) == []
    assert "document_id: $document_id" in captured["query"]
    assert captured["params"]["document_id"] == DOC_A


def test_registry_tracks_status_and_metadata(tmp_path):
    registry = DocumentRegistry(tmp_path / "documents.json")
    registry.create_document(DOC_A, "a.pdf")
    updated = registry.update_document(DOC_A, status="ready", pages_processed=2, chunks_processed=3)
    assert updated["status"] == "ready"
    assert registry.get_document(DOC_A)["chunks_processed"] == 3
    assert registry.list_documents()[0]["document_id"] == DOC_A


def test_query_rejects_missing_or_unknown_document_id():
    client = TestClient(app)
    missing = client.post("/api/query", json={"question": "hello"})
    unknown = client.post("/api/query", json={"document_id": DOC_A, "question": "hello"})
    assert missing.status_code == 422
    assert unknown.status_code == 404


def test_upload_rejects_non_pdf_cleanly():
    client = TestClient(app)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_oversized_pdf(monkeypatch):
    import backend.app.main as main_module

    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 10)
    client = TestClient(app)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("large.pdf", b"%PDF-" + b"x" * 20, "application/pdf")},
    )
    assert response.status_code == 413
