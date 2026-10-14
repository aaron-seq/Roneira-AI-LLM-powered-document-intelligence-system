"""Startup must not wait on model downloads.

``RetrievalService`` loads a sentence-transformers embedding model (~90MB) and
a cross-encoder reranker (~80MB). Both used to load inside the ASGI lifespan,
and uvicorn does not accept a single connection until the lifespan completes —
so on a cold cache nothing answered for ~25s, not even ``/api/health/live``,
whose entire job is to be the cheapest possible "the process exists" signal.
Orchestrators were entitled to conclude the container was dead, and the CI
docker job and ``docker-compose.production.yml`` both carried long start
periods to work around it.

The models now load in a background task. That buys a liveness probe that
means something, and costs a window in which the service is up but cannot
embed yet. These tests pin down what happens in that window, because the
tempting answer — fall back to keyword matching — is wrong: a lexical vector
and a model vector are points in unrelated spaces, so indexing with one and
querying with the other silently corrupts retrieval rather than degrading it.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.services.embedding_service import EmbeddingModelUnavailable
from backend.services.retrieval_service import RetrievalService


@pytest.fixture
def retrieval(workspace):
    """A retrieval service pointed at a throwaway store."""
    return RetrievalService(
        collection_name="startup_latency_test",
        persist_directory=str(workspace / "startup_vectors"),
        require_real_embeddings=False,
    )


class TestTheStoreComesUpWithoutTheModels:
    @pytest.mark.asyncio
    async def test_initialize_deferred_leaves_the_models_unloaded(self, retrieval):
        """The whole point: the caller gets control back immediately."""
        await retrieval.initialize_deferred()

        assert retrieval.is_initialized, "the vector store must be usable at once"
        assert retrieval.models_loading, "and the models must not be loaded yet"

    @pytest.mark.asyncio
    async def test_models_loading_is_true_before_the_task_even_starts(self, retrieval):
        """Readiness must not pass in the gap between deferring and starting.

        Otherwise an orchestrator routes traffic to an instance that is about
        to make every search wait for a model download.
        """
        await retrieval.initialize_deferred()

        assert retrieval.model_load_task is None, "not started yet"
        assert retrieval.models_loading, "but readiness must already say no"

    @pytest.mark.asyncio
    async def test_the_load_task_is_retained(self, retrieval):
        """A bare create_task handle can be garbage collected mid-flight.

        It is also the only place a failure can be observed from, which is how
        startup decides whether to log the keyword-only warning.
        """
        await retrieval.initialize_deferred()
        task = retrieval.start_model_load()

        assert isinstance(task, asyncio.Task)
        assert retrieval.model_load_task is task

        await task
        assert not retrieval.models_loading

    @pytest.mark.asyncio
    async def test_starting_the_load_twice_reuses_the_same_task(self, retrieval):
        await retrieval.initialize_deferred()

        first = retrieval.start_model_load()
        second = retrieval.start_model_load()

        assert first is second
        await first

    @pytest.mark.asyncio
    async def test_models_loading_is_false_when_loaded_inline(self, retrieval):
        """``initialize()`` keeps its old all-at-once meaning."""
        await retrieval.initialize()

        assert retrieval.is_initialized
        assert not retrieval.models_loading
        assert retrieval.start_model_load() is None, "nothing left to start"


class TestNothingElseUndoesTheDeferral:
    """The deferral is only worth as much as the next line of startup.

    ``DocumentProcessorService.initialize`` and ``ChatService.initialize``
    both called ``retrieval_service.initialize()`` — the full one — so the
    models were loaded immediately after the container chose to defer them.
    Every measurement looked unchanged until this was found: liveness stayed
    at ~7.4s because the load was still happening inside the lifespan.
    """

    @pytest.mark.asyncio
    async def test_the_document_processor_does_not_force_the_model_load(
        self, retrieval, workspace
    ):
        from backend.core.database import DatabaseManager
        from backend.repositories.document_repository import DocumentRepository
        from backend.services.document_processor import DocumentProcessorService
        from backend.services.local_llm_service import LocalLLMService

        await retrieval.initialize_deferred()

        manager = DatabaseManager()
        await manager.initialize()
        processor = DocumentProcessorService(
            repository=DocumentRepository(manager),
            retrieval_service=retrieval,
            llm_service=LocalLLMService(),
        )
        try:
            await processor.initialize()

            assert retrieval.models_loading, (
                "initializing the processor must not load the embedding model"
            )
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_the_chat_service_does_not_force_the_model_load(self, retrieval):
        from backend.services.chat_service import ChatService
        from backend.services.local_llm_service import LocalLLMService
        from backend.services.memory_service import MemoryService

        await retrieval.initialize_deferred()

        chat = ChatService(
            retrieval_service=retrieval,
            memory_service=MemoryService(default_context_messages=5),
            llm_service=LocalLLMService(),
        )
        await chat.initialize()

        assert retrieval.models_loading, (
            "initializing chat must not load the embedding model"
        )
        assert retrieval.is_initialized, "but the vector store must be open"


class TestRequestsWaitRatherThanDegrade:
    @pytest.mark.asyncio
    async def test_ensure_ready_waits_for_an_in_flight_load(self, retrieval):
        await retrieval.initialize_deferred()
        retrieval.start_model_load()

        await retrieval.ensure_ready()

        assert not retrieval.models_loading, "ensure_ready must not return early"

    @pytest.mark.asyncio
    async def test_ensure_ready_loads_even_if_the_task_never_started(self, retrieval):
        """A request arriving before ``start_model_load`` must still work."""
        await retrieval.initialize_deferred()

        await retrieval.ensure_ready()

        assert not retrieval.models_loading

    @pytest.mark.asyncio
    async def test_concurrent_callers_share_one_load(self, workspace, monkeypatch):
        """Ten requests arriving mid-load must not start ten model loads.

        Before the lock, ``embed_text`` did `if not initialized: initialize()`,
        so every caller that arrived during a load started another one. With
        the load moved off the startup path, that window is now reachable by
        ordinary traffic rather than only by a test.
        """
        service = RetrievalService(
            collection_name="concurrent_load_test",
            persist_directory=str(workspace / "concurrent_vectors"),
            require_real_embeddings=False,
        )

        loads = 0
        original = service.embedding_service._load

        async def counting_load() -> None:
            nonlocal loads
            loads += 1
            # Yield, so a caller that does not wait properly gets its chance
            # to start a second load.
            await asyncio.sleep(0.01)
            await original()

        monkeypatch.setattr(service.embedding_service, "_load", counting_load)

        await service.initialize_deferred()
        service.start_model_load()
        await asyncio.gather(*(service.ensure_ready() for _ in range(10)))

        assert loads == 1, f"model was loaded {loads} times"

    @pytest.mark.asyncio
    async def test_indexing_before_the_models_are_ready_still_works(self, retrieval):
        """A document uploaded during the load window is indexed correctly.

        It waits for the model rather than being indexed with lexical vectors
        that nothing will later be able to match against.
        """
        await retrieval.initialize_deferred()
        retrieval.start_model_load()

        result = await retrieval.index_document(
            document_id="during-startup",
            content="The Zephyr acquisition closed in the second quarter.",
            owner_id="tester",
        )

        assert result.success
        assert result.chunks_indexed >= 1
        assert not retrieval.models_loading


class TestTheLoadAlwaysSettles:
    """A load that fails must still end, or readiness never recovers.

    `models_loading` stays true until the load settles, and readiness returns
    503 while it is true. So an exception escaping the background task does not
    merely lose the model — it wedges `/api/health/ready` at 503 for the life
    of the process, which an orchestrator reads as "never send this traffic".
    That is worse than the keyword-only fallback it is supposed to degrade to.

    Raised in review of this change. The import previously sat at module
    scope, where a failure killed the process outright; moving it into a
    background task is what made a silent permanent 503 possible.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [
            ImportError("no module named sentence_transformers"),
            # A broken install, not an absent one. torch is exactly the kind
            # of dependency that fails this way.
            OSError("libtorch_cpu.so: cannot open shared object file"),
            RuntimeError("numpy.dtype size changed, may indicate binary incompatibility"),
        ],
        ids=["absent", "broken-shared-library", "incompatible-binary"],
    )
    async def test_any_import_failure_degrades_rather_than_hanging(
        self, workspace, monkeypatch, failure
    ):
        import backend.services.embedding_service as module

        service = RetrievalService(
            collection_name=f"import_failure_{type(failure).__name__}",
            persist_directory=str(workspace / f"import_fail_{type(failure).__name__}"),
            require_real_embeddings=False,
        )

        # Reset the cached import result, then make the import fail.
        monkeypatch.setattr(module, "_SENTENCE_TRANSFORMERS_AVAILABLE", None)
        monkeypatch.setattr(module, "_IMPORT_FAILURE", None)

        def failing_import(name: str):
            raise failure

        monkeypatch.setattr(module.importlib, "import_module", failing_import)

        await service.initialize_deferred()
        task = service.start_model_load()
        assert task is not None
        await task

        assert not service.models_loading, "readiness would be stuck at 503"
        assert not service.embeddings_are_real
        assert service.embedding_service.degraded_reason, "must say why"

    @pytest.mark.asyncio
    async def test_an_unexpected_failure_still_settles_the_load(self, workspace):
        """Belt and braces: whatever escapes, the load ends."""
        service = RetrievalService(
            collection_name="unexpected_failure_test",
            persist_directory=str(workspace / "unexpected_vectors"),
            require_real_embeddings=False,
        )

        async def explode() -> None:
            raise RuntimeError("something nobody anticipated")

        service.embedding_service.initialize = explode  # type: ignore[method-assign]

        await service.initialize_deferred()
        task = service.start_model_load()
        assert task is not None
        await task

        assert not service.models_loading


class TestRequireRealEmbeddingsIsNotDeferred:
    """A deployment that demands semantic search must still fail fast.

    ``REQUIRE_REAL_EMBEDDINGS=true`` exists so a missing model is a startup
    error rather than silent keyword-only matching. Deferring that load would
    turn the error into a background exception nobody awaits, and the process
    would come up reporting itself healthy — exactly what the setting is there
    to prevent. So the container loads inline in that case; this test pins the
    decision to the code that makes it.
    """

    @pytest.mark.asyncio
    async def test_a_required_model_that_cannot_load_fails_the_caller(
        self, workspace, monkeypatch
    ):
        """The fail-fast contract: the error reaches whoever started the app."""
        service = RetrievalService(
            collection_name="require_real_test",
            persist_directory=str(workspace / "require_real_vectors"),
            require_real_embeddings=True,
        )
        monkeypatch.setattr(
            "backend.services.embedding_service._load_sentence_transformers",
            lambda: False,
        )

        with pytest.raises(EmbeddingModelUnavailable):
            await service.initialize()

    @pytest.mark.asyncio
    async def test_startup_chooses_inline_loading_when_the_model_is_required(
        self, workspace, monkeypatch
    ):
        """``initialize_services`` must not defer under that setting.

        Deferred, the ``EmbeddingModelUnavailable`` above would surface as an
        exception on a task nobody awaits, and the process would come up
        answering health checks — which is the outcome the setting exists to
        prevent.
        """
        from backend.api import dependencies
        from backend.core.config import get_settings

        monkeypatch.setattr(
            get_settings(), "require_real_embeddings", True, raising=False
        )
        monkeypatch.setattr(
            "backend.services.embedding_service._load_sentence_transformers",
            lambda: False,
        )

        with pytest.raises(EmbeddingModelUnavailable):
            await dependencies.initialize_services()


class TestHealthReportsTheLoadWindow:
    def test_liveness_does_not_depend_on_the_models(self, client):
        """Liveness answers from the process alone — no container, no model."""
        response = client.get("/api/health/live")

        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_readiness_gates_on_the_models(self, client, app, monkeypatch):
        """503 while loading is the honest answer, and the one to route on."""
        retrieval = app.state.container.retrieval_service
        monkeypatch.setattr(type(retrieval), "models_loading", property(lambda _: True))

        response = client.get("/api/health/ready")

        assert response.status_code == 503
        assert "embedding model" in response.json()["detail"]

    def test_readiness_passes_once_loaded(self, client, auth_headers):
        """Readiness clears on its own — including when no model arrives.

        A search forces the load to completion (``ensure_ready``), which is
        also the path a real first request takes. The load settles either way:
        a model that cannot be had leaves the service keyword-only and
        *ready*, rather than failing readiness forever. Without that, a free
        tier with no model access would never pass a health check.
        """
        assert client.get("/api/health/ready").status_code in (200, 503)

        search = client.post(
            "/api/search",
            headers=auth_headers,
            json={"query": "anything at all", "top_k": 1},
        )
        assert search.status_code == 200, search.text

        assert client.get("/api/health/ready").status_code == 200

    def test_health_distinguishes_starting_from_degraded(self, client, app, monkeypatch):
        """ "Loading" is not "degraded".

        Nothing is missing during the load window; it is not all there yet.
        Reporting "degraded" on every boot would train an operator to ignore
        the one signal that says search results are keyword-only for good.

        Both properties are patched, because the real load window has both:
        the model is not loaded, and therefore not real. Patching only
        `models_loading` describes a state that cannot occur — `is_real`
        becomes true only once the load finishes — and the health check tests
        `embeddings_are_real` first, so the assertion passed here (where the
        model download is blocked) and failed in CI (where it succeeds).
        """
        retrieval = app.state.container.retrieval_service
        monkeypatch.setattr(type(retrieval), "models_loading", property(lambda _: True))
        monkeypatch.setattr(
            type(retrieval), "embeddings_are_real", property(lambda _: False)
        )

        body = client.get("/api/health").json()

        assert body["components"]["embeddings"]["status"] == "starting"
        assert body["status"] == "starting"

    def test_health_reports_degraded_once_no_model_is_coming(self, client, auth_headers):
        """With the load settled and no model, keyword-only is the end state.

        Confirms the "starting" branch has not swallowed the degraded one —
        the signal that says search results are keyword-only *for good*.
        """
        # Force the load to settle, as a first real request would.
        client.post(
            "/api/search",
            headers=auth_headers,
            json={"query": "anything at all", "top_k": 1},
        )

        embeddings = client.get("/api/health").json()["components"]["embeddings"]

        assert embeddings["status"] in ("ok", "degraded"), "the load must have settled"
        if embeddings["status"] == "degraded":
            assert "Keyword-only" in embeddings["detail"]


class TestShutdownDuringStartup:
    @pytest.mark.asyncio
    async def test_cleanup_cancels_an_in_flight_load(self, workspace):
        """A container killed mid-boot must not leave a task running.

        The load holds the services cleanup is about to tear down.
        """
        service = RetrievalService(
            collection_name="shutdown_during_load_test",
            persist_directory=str(workspace / "shutdown_vectors"),
            require_real_embeddings=False,
        )

        never_finishes = asyncio.Event()

        async def hang() -> None:
            await never_finishes.wait()

        service.embedding_service._load = hang  # type: ignore[method-assign]

        await service.initialize_deferred()
        task = service.start_model_load()
        assert task is not None
        await asyncio.sleep(0)  # let the task start
        await service.cleanup()

        assert task.cancelled() or task.done()
        assert service.model_load_task is None
