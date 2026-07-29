"""Application service container and FastAPI dependency providers.

Services are constructed once at startup and wired together explicitly. The
important relationship: ``DocumentProcessorService`` and ``ChatService`` share
one ``RetrievalService``. When each built its own, uploads went into one vector
store and chat searched a different, empty one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, status
from starlette.requests import HTTPConnection

from backend.core.config import get_settings
from backend.core.database import DatabaseManager
from backend.repositories.document_repository import DocumentRepository
from backend.repositories.feedback_repository import FeedbackRepository
from backend.services.auth_service import AuthService
from backend.services.chat_service import ChatService
from backend.services.document_processor import DocumentProcessorService
from backend.services.guardrail_service import GuardrailService
from backend.services.local_llm_service import LocalLLMService
from backend.services.memory_service import MemoryService
from backend.services.retrieval_service import RetrievalService
from backend.services.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


@dataclass
class ServiceContainer:
    """Holds the singleton services for the running application."""

    db_manager: DatabaseManager
    document_repository: DocumentRepository
    feedback_repository: FeedbackRepository
    retrieval_service: RetrievalService
    llm_service: LocalLLMService
    document_processor: DocumentProcessorService
    chat_service: ChatService
    auth_service: AuthService
    guardrail_service: GuardrailService
    websocket_manager: WebSocketManager


async def initialize_services(app=None) -> ServiceContainer:
    """Construct and initialize every application service.

    Args:
        app: The FastAPI application to attach the container to. The container
            is stored on ``app.state`` rather than in a module global so two
            application instances in one process (tests, embedded usage) do
            not share — or tear down — each other's services.

    Returns:
        The populated service container.

    Raises:
        Exception: if a service the application cannot run without fails to
            start. Optional capabilities (the LLM) degrade instead.
    """
    settings = get_settings()

    db_manager = DatabaseManager()
    await db_manager.initialize()

    document_repository = DocumentRepository(db_manager)
    feedback_repository = FeedbackRepository(db_manager)

    # One retrieval service, one embedding model, one vector store.
    retrieval_service = RetrievalService()
    await retrieval_service.initialize()

    # One LLM client shared by document enrichment and chat, so a single
    # connection pool and one model-availability check serve both.
    llm_service = LocalLLMService()

    document_processor = DocumentProcessorService(
        repository=document_repository,
        retrieval_service=retrieval_service,
        llm_service=llm_service,
    )
    await document_processor.initialize()

    chat_service = ChatService(
        retrieval_service=retrieval_service,
        memory_service=MemoryService(
            default_context_messages=settings.max_conversation_history
        ),
        llm_service=llm_service,
    )
    await chat_service.initialize()

    container = ServiceContainer(
        db_manager=db_manager,
        document_repository=document_repository,
        feedback_repository=feedback_repository,
        retrieval_service=retrieval_service,
        llm_service=llm_service,
        document_processor=document_processor,
        chat_service=chat_service,
        auth_service=AuthService(),
        guardrail_service=GuardrailService(),
        websocket_manager=WebSocketManager(),
    )

    if app is not None:
        app.state.container = container

    _publish_startup_metrics(settings, retrieval_service)
    return container


def _publish_startup_metrics(settings, retrieval_service: RetrievalService) -> None:
    """Seed the metrics registry with build and capability information."""
    try:
        from backend.observability.metrics import get_metrics

        metrics = get_metrics()
        metrics.set_app_info(version=settings.version, environment=settings.environment)
        metrics.set_embedding_backend_real(retrieval_service.embeddings_are_real)
    except Exception as exc:  # pragma: no cover - metrics are best-effort
        logger.debug("Could not publish startup metrics: %s", exc)


async def cleanup_services(app=None) -> None:
    """Release every resource held by ``app``'s container."""
    container: Optional[ServiceContainer] = getattr(
        getattr(app, "state", None), "container", None
    )
    if container is None:
        return

    for name, close in (
        ("chat", container.chat_service.cleanup),
        ("llm", container.llm_service.close),
        ("database", container.db_manager.close),
    ):
        try:
            await close()
        except Exception as exc:
            logger.warning("Error shutting down %s service: %s", name, exc)

    app.state.container = None


def get_container(connection: HTTPConnection) -> ServiceContainer:
    """Return the container belonging to the application handling this request.

    ``HTTPConnection`` is the shared base of ``Request`` and ``WebSocket``, so
    this one dependency serves both HTTP routes and websocket endpoints.

    Raises:
        HTTPException: 503 if the application has not finished starting, which
            is the honest answer to a request that arrives too early.
    """
    container = getattr(connection.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is starting up. Retry in a moment.",
        )
    return container


# --------------------------------------------------------------------------
# FastAPI dependency providers
# --------------------------------------------------------------------------


def get_document_processor(
    container: ServiceContainer = Depends(get_container),
) -> DocumentProcessorService:
    return container.document_processor


def get_chat_service(
    container: ServiceContainer = Depends(get_container),
) -> ChatService:
    return container.chat_service


def get_auth_service(
    container: ServiceContainer = Depends(get_container),
) -> AuthService:
    return container.auth_service


def get_guardrail_service(
    container: ServiceContainer = Depends(get_container),
) -> GuardrailService:
    return container.guardrail_service


def get_websocket_manager(
    container: ServiceContainer = Depends(get_container),
) -> WebSocketManager:
    return container.websocket_manager


def get_db_manager(
    container: ServiceContainer = Depends(get_container),
) -> DatabaseManager:
    return container.db_manager


def get_document_repository(
    container: ServiceContainer = Depends(get_container),
) -> DocumentRepository:
    return container.document_repository


def get_feedback_repository(
    container: ServiceContainer = Depends(get_container),
) -> FeedbackRepository:
    return container.feedback_repository


def get_retrieval_service(
    container: ServiceContainer = Depends(get_container),
) -> RetrievalService:
    return container.retrieval_service
