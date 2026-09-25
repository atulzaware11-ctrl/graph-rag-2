"""Small JSON-backed registry for locally indexed documents."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


class DocumentRegistry:
    """Keep document metadata separate from the indexing services."""

    def __init__(self, registry_path: Path | None = None) -> None:
        self._documents: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        default_path = Path(__file__).resolve().parents[2] / "data" / "documents.json"
        self._path = registry_path or Path(os.getenv("DOCUMENT_REGISTRY_PATH", default_path))
        self._load()

    def _load(self) -> None:
        try:
            self._documents = {
                item["document_id"]: item
                for item in json.loads(self._path.read_text(encoding="utf-8"))
            }
        except FileNotFoundError:
            self._documents = {}
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            self._documents = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(list(self._documents.values()), indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, self._path)

    def create_document(self, document_id: str, filename: str) -> dict[str, Any]:
        document = {
            "document_id": document_id,
            "filename": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pages_processed": 0,
            "chunks_processed": 0,
            "status": "uploading",
        }
        with self._lock:
            self._documents[document_id] = document
            self._save()
            return document.copy()

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._lock:
            document = self._documents.get(document_id)
            return document.copy() if document else None

    def update_document(self, document_id: str, **updates: Any) -> dict[str, Any] | None:
        with self._lock:
            document = self._documents.get(document_id)
            if document is None:
                return None
            document.update(updates)
            self._save()
            return document.copy()

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return [document.copy() for document in self._documents.values()]


document_registry = DocumentRegistry()
