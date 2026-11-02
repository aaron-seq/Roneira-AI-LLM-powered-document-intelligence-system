"""Regression tests for authorization before vector candidate limiting."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.services.retrieval_service import RetrievalService
from backend.services.vector_store_service import InMemoryVectorStore, VectorDocument


@dataclass
class _Embedding:
    embedding: list[float]


class _EmbeddingService:
    def __init__(self, *, is_real: bool):
        self.dimension = 2
        self.is_real = is_real
        self.model_name = "test-model"
        self.degraded_reason = None if is_real else "test fallback"

    async def initialize(self):
        return None

    async def embed_text(self, text: str):
        return _Embedding([1.0, 0.0])


class _NoReranker:
    async def initialize(self):
        return None

    async def rank(self, _query, _documents):
        return None


class _VectorStore:
    def __init__(self):
        self.is_initialized = True
        self._memory_store = InMemoryVectorStore()
        self.last_top_k = None
        self.last_filter = None

    async def search(self, query_embedding, top_k, filter_dict=None):
        self.last_top_k = top_k
        self.last_filter = filter_dict
        return self._memory_store.search(query_embedding, top_k, filter_dict)


def _service(*, is_real: bool) -> RetrievalService:
    service = RetrievalService.__new__(RetrievalService)
    service.embedding_service = _EmbeddingService(is_real=is_real)
    service.vector_store = _VectorStore()
    service.reranker = _NoReranker()
    service.hybrid_enabled = False
    service.min_score = 0.0
    service.is_initialized = True
    service._models_loaded = False
    service._models_deferred = False
    service._model_load_task = None
    service.chunk_size = 1000
    service.chunk_overlap = 200
    service.persist_directory = None
    return service


def _add_documents(service: RetrievalService, foreign_count: int = 100) -> None:
    documents = [
        VectorDocument(
            chunk_id="owned_chunk",
            document_id="owned-document",
            content="owned cobalt anchor policy",
            embedding=[0.8, 0.6],
            metadata={"owner_id": "owner-a"},
        )
    ]
    documents.extend(
        VectorDocument(
            chunk_id=f"foreign_chunk_{index}",
            document_id=f"foreign-document-{index}",
            content="foreign cobalt anchor policy",
            embedding=[1.0, 0.0],
            metadata={"owner_id": "owner-b"},
        )
        for index in range(foreign_count)
    )
    service.vector_store._memory_store.add_documents(documents)


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_count", [6, 100])
@pytest.mark.parametrize("is_real", [False, True])
async def test_authorized_document_survives_foreign_candidates(
    is_real: bool, foreign_count: int
):
    service = _service(is_real=is_real)
    _add_documents(service, foreign_count=foreign_count)

    result = await service.retrieve(
        query="cobalt anchor",
        top_k=1,
        min_score=0.0,
        owner_id="owner-a",
        document_ids=["owned-document"],
    )

    assert [item.document_id for item in result.results] == ["owned-document"]
    assert result.results[0].metadata["owner_id"] == "owner-a"
    assert service.vector_store.last_top_k == 4
    assert service.vector_store.last_filter == {
        "document_id": {"$in": ["owned-document"]}
    }


@pytest.mark.asyncio
async def test_authorized_document_ids_keep_legacy_ownerless_chunks():
    service = _service(is_real=False)
    service.vector_store._memory_store.add_documents(
        [
            VectorDocument(
                chunk_id="legacy_chunk",
                document_id="legacy-document",
                content="legacy cobalt anchor policy",
                embedding=[1.0, 0.0],
                metadata={},
            ),
            VectorDocument(
                chunk_id="foreign_chunk",
                document_id="foreign-document",
                content="foreign cobalt anchor policy",
                embedding=[0.99, 0.01],
                metadata={"owner_id": "owner-b"},
            ),
        ]
    )

    result = await service.retrieve(
        query="cobalt anchor",
        top_k=1,
        min_score=0.0,
        owner_id="owner-a",
        document_ids=["legacy-document"],
    )

    assert [item.document_id for item in result.results] == ["legacy-document"]
