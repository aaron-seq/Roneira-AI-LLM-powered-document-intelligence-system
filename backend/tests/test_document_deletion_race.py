from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from sqlalchemy import func, select

from backend.core.models import DocumentChunk


def observe_processing(monkeypatch: pytest.MonkeyPatch, processor: Any) -> dict[str, str]:
    captured: dict[str, str] = {}
    original_process = processor.process_document

    async def observe_process(*args: Any, **kwargs: Any) -> None:
        document_id = kwargs.get("document_id")
        if document_id is None:
            document_id = args[0]
        captured["document_id"] = document_id
        await original_process(*args, **kwargs)

    monkeypatch.setattr(processor, "process_document", observe_process)
    return captured


def stored_source_path(client, container, document_id: str) -> str:
    document = client.portal.call(container.document_repository.get, document_id)
    assert document is not None
    assert document.storage_path is not None
    return document.storage_path


def chunk_count(client, container, document_id: str) -> int:
    async def read_count() -> int:
        async with container.db_manager.get_session() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
            )
            return int(count or 0)

    return client.portal.call(read_count)


def vector_count(client, container, document_id: str) -> int:
    async def read_count() -> int:
        vector_store = container.retrieval_service.vector_store
        if vector_store._use_chromadb and vector_store._collection is not None:
            collection = vector_store._collection
            vectors = await asyncio.to_thread(
                collection.get,
                where={"document_id": document_id},
                include=["metadatas"],
            )
            return len(vectors["ids"])

        memory_store = vector_store._memory_store
        if memory_store is None:
            return 0
        return sum(
            1
            for vector in memory_store.documents.values()
            if vector.document_id == document_id
        )

    return client.portal.call(read_count)


def upload(client, auth_headers):
    return client.post(
        "/api/documents/upload",
        headers=auth_headers,
        files={
            "file": (
                "delete-race.txt",
                b"A deterministic document used to verify deletion during indexing.",
                "text/plain",
            )
        },
    )


def assert_deleted_state(client, auth_headers, container, document_id, source_path):
    assert (
        client.get(
            f"/api/documents/{document_id}/status", headers=auth_headers
        ).status_code
        == 404
    )
    assert chunk_count(client, container, document_id) == 0
    assert vector_count(client, container, document_id) == 0
    assert not Path(source_path).exists()


def test_delete_during_enrichment_prevents_stale_indexing(
    monkeypatch, client, app, auth_headers
) -> None:
    container = app.state.container
    processor = container.document_processor
    enrichment_started = Event()
    allow_enrichment = Event()

    async def pause_enrichment(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        enrichment_started.set()
        if not await asyncio.to_thread(allow_enrichment.wait, 30):
            raise TimeoutError("enrichment gate was not released")
        return {}

    monkeypatch.setattr(processor, "_enrich", pause_enrichment)
    captured = observe_processing(monkeypatch, processor)

    with ThreadPoolExecutor(max_workers=1) as executor:
        upload_future = executor.submit(upload, client, auth_headers)
        try:
            assert enrichment_started.wait(15)
            document_id = captured["document_id"]
            source_path = stored_source_path(client, container, document_id)
            response = client.delete(
                f"/api/documents/{document_id}", headers=auth_headers
            )
        finally:
            allow_enrichment.set()

        upload_response = upload_future.result(timeout=30)

    assert response.status_code == 200
    assert upload_response.status_code == 202
    assert processor._document_locks == {}
    assert_deleted_state(client, auth_headers, container, document_id, source_path)


def test_delete_waits_for_indexing_and_persistence(
    monkeypatch, client, app, auth_headers
) -> None:
    container = app.state.container
    processor = container.document_processor
    retrieval = container.retrieval_service
    index_started = Event()
    allow_index = Event()
    delete_started = Event()
    delete_finished = Event()
    original_index = retrieval.index_document
    original_delete = processor.delete_document

    async def pause_index(**kwargs: Any):
        index_started.set()
        if not await asyncio.to_thread(allow_index.wait, 30):
            raise TimeoutError("index gate was not released")
        return await original_index(**kwargs)

    async def observe_delete(document_id: str, owner_id: str) -> bool:
        delete_started.set()
        try:
            return await original_delete(document_id, owner_id)
        finally:
            delete_finished.set()

    monkeypatch.setattr(retrieval, "index_document", pause_index)
    monkeypatch.setattr(processor, "delete_document", observe_delete)
    captured = observe_processing(monkeypatch, processor)

    with ThreadPoolExecutor(max_workers=2) as executor:
        upload_future = executor.submit(upload, client, auth_headers)
        delete_future = None
        try:
            assert index_started.wait(15)
            document_id = captured["document_id"]
            source_path = stored_source_path(client, container, document_id)
            delete_future = executor.submit(
                client.delete,
                f"/api/documents/{document_id}",
                headers=auth_headers,
            )
            assert delete_started.wait(5)
            deletion_waited_for_index = not delete_finished.wait(0.2)
        finally:
            allow_index.set()

        upload_response = upload_future.result(timeout=30)
        assert delete_future is not None
        delete_response = delete_future.result(timeout=30)

    assert deletion_waited_for_index
    assert upload_response.status_code == 202
    assert delete_response.status_code == 200
    assert processor._document_locks == {}
    assert_deleted_state(client, auth_headers, container, document_id, source_path)


def test_sql_chunk_failure_removes_newly_written_vectors(
    monkeypatch, client, app, auth_headers
) -> None:
    container = app.state.container
    repository = container.document_repository

    async def fail_chunk_persistence(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("injected chunk persistence failure")

    monkeypatch.setattr(repository, "replace_chunks", fail_chunk_persistence)
    captured = observe_processing(monkeypatch, container.document_processor)

    response = upload(client, auth_headers)
    document_id = captured["document_id"]

    assert response.status_code == 202
    status = client.get(f"/api/documents/{document_id}/status", headers=auth_headers)
    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert chunk_count(client, container, document_id) == 0
    assert vector_count(client, container, document_id) == 0
    assert container.document_processor._document_locks == {}
