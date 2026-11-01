"""Retrieval quality and answer grounding.

Unit-level tests for the pieces that determine whether an answer can be
trusted: chunk-to-page mapping for citations, score thresholding, context
assembly, and the honesty signals attached to every response.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from backend.services.embedding_service import (
    EmbeddingModelUnavailable,
    EmbeddingService,
)
from backend.services.retrieval_service import (
    RetrievalService,
    _page_for_offset,
    _page_offsets,
    _rank_hybrid,
    _tokenise,
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


class TestHybridRanking:
    """Keyword rank must rescue the chunk holding the literal term asked for.

    Vector search ranks by meaning, so a chunk *about* invoices can outrank the
    one containing "INV-2025-1001". That is the case people notice, because an
    identifier is exactly what they searched for.
    """

    @staticmethod
    def _result(chunk_id, content, score):
        return SearchResult(
            chunk_id=chunk_id,
            document_id="doc1",
            content=content,
            score=score,
            metadata={},
        )

    def test_the_chunk_with_the_exact_identifier_is_promoted(self):
        # Vector order puts the generic prose first; only the third chunk
        # actually contains the identifier.
        candidates = [
            self._result("a", "This invoice covers professional services.", 0.71),
            self._result("b", "Invoices are issued monthly to each client.", 0.70),
            self._result("c", "Invoice INV-2025-1001 total 12480.00 USD.", 0.69),
        ]

        ranked = _rank_hybrid("what is INV-2025-1001", candidates)

        assert ranked[0].chunk_id == "c"

    def test_scores_are_left_alone(self):
        """Reordering must not rewrite the similarity a citation reports."""
        candidates = [
            self._result("a", "Invoices are issued monthly.", 0.70),
            self._result("b", "Invoice INV-2025-1001 total due.", 0.69),
        ]

        ranked = _rank_hybrid("INV-2025-1001", candidates)

        assert {r.chunk_id: r.score for r in ranked} == {"a": 0.70, "b": 0.69}

    def test_a_query_matching_no_candidate_keeps_vector_order(self):
        candidates = [
            self._result("a", "Revenue grew year over year.", 0.8),
            self._result("b", "Headcount was unchanged.", 0.7),
        ]

        ranked = _rank_hybrid("zzz nonexistent term", candidates)

        assert [r.chunk_id for r in ranked] == ["a", "b"]

    def test_identifiers_survive_tokenisation(self):
        """Splitting on '-' would make INV-2025-1001 match any invoice."""
        assert "inv-2025-1001" in _tokenise("Invoice INV-2025-1001 issued")

    def test_a_term_in_every_candidate_does_not_decide_the_order(self):
        """A term with no discriminating power must not drive ranking."""
        candidates = [
            self._result("a", "invoice one", 0.9),
            self._result("b", "invoice two", 0.8),
        ]

        ranked = _rank_hybrid("invoice", candidates)

        assert [r.chunk_id for r in ranked] == ["a", "b"]


class TestChatGrounding:
    """Answers must declare whether they rest on retrieved sources."""

    def test_chat_route_forwards_turn_id(self, app, client, admin_headers):
        from backend.api.dependencies import get_chat_service

        captured = {}

        class FakeChatService:
            async def chat(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    message="answer",
                    session_id=kwargs["session_id"],
                    sources=[],
                    model="fake",
                    usage={},
                )

        previous = app.dependency_overrides.get(get_chat_service)
        app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
        try:
            response = client.post(
                "/api/chat",
                headers=admin_headers,
                json={
                    "message": "Q1",
                    "session_id": "session-api",
                    "turn_id": "turn-api",
                    "use_rag": False,
                },
            )
        finally:
            if previous is None:
                app.dependency_overrides.pop(get_chat_service, None)
            else:
                app.dependency_overrides[get_chat_service] = previous

        assert response.status_code == 200
        assert captured["turn_id"] == "turn-api"

    def test_chat_route_preserves_service_unavailable_status(
        self, app, client, admin_headers
    ):
        from backend.api.dependencies import get_chat_service
        from backend.utils.exceptions import LLMUnavailableError

        class UnavailableChatService:
            async def chat(self, **_):
                raise LLMUnavailableError("The language model is unavailable.")

        previous = app.dependency_overrides.get(get_chat_service)
        app.dependency_overrides[get_chat_service] = lambda: UnavailableChatService()
        try:
            response = client.post(
                "/api/chat",
                headers=admin_headers,
                json={
                    "message": "Q1",
                    "session_id": "session-api",
                    "turn_id": "turn-api",
                    "use_rag": False,
                },
            )
        finally:
            if previous is None:
                app.dependency_overrides.pop(get_chat_service, None)
            else:
                app.dependency_overrides[get_chat_service] = previous

        assert response.status_code == 503
        assert response.json()["detail"] == "The language model is unavailable."

    @staticmethod
    def _service_with_answers(answers):
        from backend.services.chat_service import ChatService
        from backend.services.memory_service import MemoryService
        from backend.services.prompt_service import PromptService

        class Retrieval:
            embeddings_are_real = True

        class FakeLLM:
            settings = SimpleNamespace(ollama_model="fake")

            def __init__(self):
                self.answers = iter(answers)
                self.prompts = []

            async def generate_chat_response(self, prompt):
                self.prompts.append(prompt)
                answer = next(self.answers)
                if isinstance(answer, BaseException):
                    raise answer
                return answer

        memory = MemoryService()
        llm = FakeLLM()
        service = ChatService(
            retrieval_service=Retrieval(),
            memory_service=memory,
            prompt_service=PromptService(),
            llm_service=llm,
        )
        service.is_initialized = True
        return service, memory, llm

    @pytest.mark.asyncio
    async def test_regenerating_an_earlier_turn_branches_memory_and_context(self):
        service, memory, llm = self._service_with_answers(
            ["A1 old", "A2", "A1 replacement", "A3"]
        )

        await service.chat("Q1", "session1", use_rag=False, turn_id="turn-1")
        await service.chat("Q2", "session1", use_rag=False, turn_id="turn-2")
        await service.chat("Q1", "session1", use_rag=False, turn_id="turn-1")

        regeneration_prompt = llm.prompts[2]
        assert "A1 old" not in regeneration_prompt
        assert "Q2" not in regeneration_prompt
        assert "A2" not in regeneration_prompt
        history = await memory.get_conversation_history("session1")
        assert [(item["role"], item["content"]) for item in history] == [
            ("user", "Q1"),
            ("assistant", "A1 replacement"),
        ]

        await service.chat("Q3", "session1", use_rag=False, turn_id="turn-3")
        next_prompt = llm.prompts[3]
        assert "A1 replacement" in next_prompt
        assert "A1 old" not in next_prompt
        assert "Q2" not in next_prompt
        assert "A2" not in next_prompt

    @pytest.mark.asyncio
    async def test_regenerating_the_last_turn_preserves_earlier_context(self):
        service, memory, llm = self._service_with_answers(["A1", "A2 old", "A2 new"])

        await service.chat("Q1", "session1", use_rag=False, turn_id="turn-1")
        await service.chat("Q2", "session1", use_rag=False, turn_id="turn-2")
        await service.chat("Q2", "session1", use_rag=False, turn_id="turn-2")

        regeneration_prompt = llm.prompts[2]
        assert "A1" in regeneration_prompt
        assert "A2 old" not in regeneration_prompt
        history = await memory.get_conversation_history("session1")
        assert [(item["role"], item["content"]) for item in history] == [
            ("user", "Q1"),
            ("assistant", "A1"),
            ("user", "Q2"),
            ("assistant", "A2 new"),
        ]

    @pytest.mark.asyncio
    async def test_failed_regeneration_restores_the_original_history(self):
        from backend.utils.exceptions import LLMUnavailableError

        service, memory, _ = self._service_with_answers(
            [
                "A1",
                "A2",
                LLMUnavailableError("The language model is unavailable."),
            ]
        )

        await service.chat("Q1", "session1", use_rag=False, turn_id="turn-1")
        await service.chat("Q2", "session1", use_rag=False, turn_id="turn-2")
        before = await memory.get_conversation_history("session1")

        with pytest.raises(LLMUnavailableError, match="unavailable"):
            await service.chat("Q1", "session1", use_rag=False, turn_id="turn-1")

        assert await memory.get_conversation_history("session1") == before

    @pytest.mark.asyncio
    async def test_concurrent_turns_in_one_session_are_serialized(self):
        from backend.services.chat_service import ChatService
        from backend.services.memory_service import MemoryService
        from backend.services.prompt_service import PromptService

        class Retrieval:
            embeddings_are_real = True

        class GatedLLM:
            settings = SimpleNamespace(ollama_model="fake")

            def __init__(self):
                self.first_started = asyncio.Event()
                self.release_first = asyncio.Event()
                self.prompts = []

            async def generate_chat_response(self, prompt):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    self.first_started.set()
                    await self.release_first.wait()
                    return "A1"
                return "A2"

        memory = MemoryService()
        llm = GatedLLM()
        service = ChatService(
            retrieval_service=Retrieval(),
            memory_service=memory,
            prompt_service=PromptService(),
            llm_service=llm,
        )
        service.is_initialized = True

        first = asyncio.create_task(
            service.chat("Q1", "session1", use_rag=False, turn_id="turn-1")
        )
        await llm.first_started.wait()
        second = asyncio.create_task(
            service.chat("Q2", "session1", use_rag=False, turn_id="turn-2")
        )
        await asyncio.sleep(0)
        assert len(llm.prompts) == 1

        llm.release_first.set()
        await asyncio.gather(first, second)

        assert "A1" in llm.prompts[1]
        history = await memory.get_conversation_history("session1")
        assert [(item["role"], item["content"]) for item in history] == [
            ("user", "Q1"),
            ("assistant", "A1"),
            ("user", "Q2"),
            ("assistant", "A2"),
        ]

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

    @pytest.mark.asyncio
    async def test_a_failed_generation_is_never_reported_as_grounded(self):
        """Retrieval succeeding is not the same as an answer being produced.

        The LLM being down used to return an apology *string*, which chat
        happily shipped as the answer with ``grounded: true`` and a full
        citation list — a "grounded" badge next to a failure notice.
        """
        from backend.services.chat_service import ChatService
        from backend.utils.exceptions import LLMUnavailableError

        hit = SearchResult(
            chunk_id="doc1_chunk_0",
            document_id="doc1",
            content="Total amount due: $12,480.00",
            score=0.9,
            metadata={"filename": "invoice.pdf", "page_number": 1},
        )

        class Retrieval:
            embeddings_are_real = True

            async def retrieve(self, **_):
                return SimpleNamespace(results=[hit], combined_context="ctx")

        class Memory:
            @asynccontextmanager
            async def session_transaction(self, *_):
                yield SimpleNamespace(messages=[])

            async def add_user_message(self, *_, **__): ...
            async def add_assistant_message(self, *_, **__): ...
            async def get_context_messages(self, *_):
                return []

        class Prompts:
            def build_chat_messages(self, **_):
                return [{"role": "user", "content": "q"}]

        class DeadLLM:
            settings = SimpleNamespace(ollama_model="llama3.2:3b")

            async def generate_chat_response(self, _prompt):
                raise LLMUnavailableError("The language model is unavailable.")

        service = ChatService(
            retrieval_service=Retrieval(),
            memory_service=Memory(),
            prompt_service=Prompts(),
            llm_service=DeadLLM(),
        )
        service.is_initialized = True

        response = await service.chat(
            "What is the invoice total?", turn_id="new-turn", use_rag=True
        )

        assert response.usage["grounded"] is False, (
            "generation failed, so no answer was built from the passages"
        )
        # The retrieved passages are still worth showing — the user can read
        # them directly — but they must not be presented as a cited answer.
        assert response.sources, "relevant passages should still be surfaced"
        assert "unavailable" in response.message.lower()

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_leak_internals_into_the_answer(self):
        """Transport errors carry the configured endpoint URL."""
        from backend.services.chat_service import ChatService
        from backend.utils.exceptions import LLMUnavailableError

        class Retrieval:
            embeddings_are_real = True

            async def retrieve(self, **_):
                return SimpleNamespace(results=[], combined_context="")

        class Memory:
            @asynccontextmanager
            async def session_transaction(self, *_):
                yield SimpleNamespace(messages=[])

            async def add_user_message(self, *_, **__): ...
            async def add_assistant_message(self, *_, **__): ...
            async def get_context_messages(self, *_):
                return []

        class Prompts:
            def build_chat_messages(self, **_):
                return [{"role": "user", "content": "q"}]

        class LeakyLLM:
            settings = SimpleNamespace(ollama_model="llama3.2:3b")

            async def generate_chat_response(self, _prompt):
                raise LLMUnavailableError(
                    "The language model failed while generating a response.",
                    original_error=RuntimeError(
                        "connection refused to http://internal-host:11434"
                    ),
                )

        service = ChatService(
            retrieval_service=Retrieval(),
            memory_service=Memory(),
            prompt_service=Prompts(),
            llm_service=LeakyLLM(),
        )
        service.is_initialized = True

        message = (await service.chat("hello")).message
        assert "internal-host" not in message
        assert "11434" not in message

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
