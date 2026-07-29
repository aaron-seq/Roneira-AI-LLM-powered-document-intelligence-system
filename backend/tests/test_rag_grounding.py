"""Retrieval quality and answer grounding.

Unit-level tests for the pieces that determine whether an answer can be
trusted: chunk-to-page mapping for citations, score thresholding, context
assembly, and the honesty signals attached to every response.
"""

from __future__ import annotations

import pytest

from backend.services.embedding_service import (
    EmbeddingModelUnavailable,
    EmbeddingService,
)
from backend.services.retrieval_service import (
    RetrievalService,
    _page_for_offset,
    _page_offsets,
)
from backend.services.vector_store_service import SearchResult


class TestPageCitations:
    """Chunks must map back to a page a human can open and check."""

    SAMPLE = (
        "--- Page 1 ---\nIntroduction and scope.\n"
        "--- Page 2 ---\nRevenue was 4.2 million.\n"
        "--- Page 3 ---\nAppendix and definitions.\n"
    )

    def test_page_markers_are_located(self):
        offsets = _page_offsets(self.SAMPLE)
        assert [page for _, page in offsets] == [1, 2, 3]

    def test_offset_maps_to_the_containing_page(self):
        offsets = _page_offsets(self.SAMPLE)
        revenue_position = self.SAMPLE.index("Revenue")
        assert _page_for_offset(offsets, revenue_position) == 2

    def test_offset_before_any_marker_has_no_page(self):
        assert _page_for_offset(_page_offsets(self.SAMPLE), 0) == 1
        assert _page_for_offset([], 42) is None

    def test_unpaginated_text_yields_no_page(self):
        assert _page_offsets("Plain text with no page markers.") == []


class TestContextAssembly:
    """Retrieved context is fed to an LLM, so its shape matters."""

    @staticmethod
    def _result(content: str, page: int | None = None, name: str = "report.pdf"):
        metadata = {"filename": name}
        if page is not None:
            metadata["page_number"] = page
        return SearchResult(
            document_id="doc-1",
            chunk_id="doc-1_chunk_0",
            content=content,
            score=0.9,
            metadata=metadata,
        )

    def test_each_chunk_is_labelled_with_its_source(self):
        service = RetrievalService()
        context = service._combine_context([self._result("Revenue rose.", page=2)], 4000)
        assert "[Source 1: report.pdf, page 2]" in context
        assert "Revenue rose." in context

    def test_context_respects_the_length_budget(self):
        """An overlong context silently truncates the prompt at the model."""
        service = RetrievalService()
        results = [self._result("x" * 500) for _ in range(20)]
        context = service._combine_context(results, max_length=1000)
        assert len(context) <= 1200  # budget plus per-chunk headers

    def test_no_results_produce_empty_context(self):
        assert RetrievalService()._combine_context([], 4000) == ""

    def test_sources_are_numbered_so_an_answer_can_reference_them(self):
        service = RetrievalService()
        context = service._combine_context(
            [self._result("First."), self._result("Second.")], 4000
        )
        assert "[Source 1:" in context and "[Source 2:" in context


class TestLexicalFallback:
    """The no-model fallback must still perform real keyword matching."""

    @pytest.fixture
    def service(self):
        return EmbeddingService(model_name="all-MiniLM-L6-v2", use_cache=False)

    def test_identical_text_scores_near_one(self, service):
        text = "quarterly revenue increased by eighteen percent"
        a = service._generate_mock_embedding(text)
        b = service._generate_mock_embedding(text)
        assert _cosine(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_shared_keywords_score_higher_than_unrelated_text(self, service):
        query = service._generate_mock_embedding("quarterly revenue growth")
        relevant = service._generate_mock_embedding(
            "The quarterly revenue growth was strong this period."
        )
        unrelated = service._generate_mock_embedding(
            "Kitchen renovation permits require council approval."
        )
        assert _cosine(query, relevant) > _cosine(query, unrelated)

    def test_vectors_are_unit_length(self, service):
        vector = service._generate_mock_embedding("some representative text")
        assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-6)

    def test_empty_text_yields_a_zero_vector(self, service):
        assert not any(service._generate_mock_embedding("   "))

    def test_vector_has_the_declared_dimension(self, service):
        assert len(service._generate_mock_embedding("text")) == service.dimension


class TestDegradedModeIsDeclared:
    """A caller must always be able to tell which backend answered."""

    @pytest.mark.asyncio
    async def test_fallback_is_reported_as_not_real(self):
        service = EmbeddingService(use_cache=False)
        await service.initialize()
        if service.is_real:
            pytest.skip("a real embedding model is installed")

        assert service.backend == "lexical-fallback"
        assert service.degraded_reason
        assert service.get_cache_stats()["embeddings_are_real"] is False

    @pytest.mark.asyncio
    async def test_require_real_model_fails_loudly(self):
        """Production must be able to refuse the fallback outright."""
        service = EmbeddingService(use_cache=False, require_real_model=True)
        try:
            await service.initialize()
        except EmbeddingModelUnavailable as exc:
            assert "REQUIRE_REAL_EMBEDDINGS" in str(exc)
        else:
            assert service.is_real, "initialize() succeeded without a real model"


class TestChatGrounding:
    """Answers must declare whether they rest on retrieved sources."""

    def test_chat_without_indexed_documents_is_not_grounded(self, client, admin_headers):
        response = client.post(
            "/api/chat",
            headers=admin_headers,
            json={"message": "What does the Zephyr contract say?", "use_rag": True},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is False
        assert body["sources"] == []

    def test_an_ungrounded_answer_says_so_in_the_text(self, client, admin_headers):
        """Silently answering from parametric memory is how RAG products lie."""
        response = client.post(
            "/api/chat",
            headers=admin_headers,
            json={"message": "Summarise the acquisition terms.", "use_rag": True},
        )
        assert "could not find" in response.json()["message"].lower()

    def test_responses_declare_the_embedding_backend(self, client, admin_headers):
        response = client.post(
            "/api/chat", headers=admin_headers, json={"message": "hello"}
        )
        assert "embeddings_are_real" in response.json()

    def test_a_session_id_is_always_returned(self, client, admin_headers):
        """Clients need it to continue the conversation."""
        response = client.post(
            "/api/chat", headers=admin_headers, json={"message": "hello"}
        )
        assert response.json()["session_id"]


def _cosine(a, b) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    magnitude = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / magnitude if magnitude else 0.0
