"""Root, health, readiness, metrics, feedback and dashboard endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from backend.api.dependencies import (
    get_db_manager,
    get_document_processor,
    get_feedback_repository,
    get_retrieval_service,
)
from backend.api.security import CurrentUser, get_current_user
from backend.core.config import get_settings
from backend.core.database import DatabaseManager
from backend.models.responses import ComponentHealth, HealthCheckResponse
from backend.repositories.feedback_repository import FeedbackRepository
from backend.services.document_processor import DocumentProcessorService
from backend.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    """A thumbs up/down on an assistant answer."""

    message_id: str = Field(..., min_length=1)
    is_positive: bool
    session_id: Optional[str] = None
    comment: Optional[str] = Field(default=None, max_length=2000)


class DashboardMetricsResponse(BaseModel):
    """Aggregate figures for the dashboard.

    Every field is measured. The previous implementation returned a
    hard-coded 98.5% average confidence whenever no document carried one,
    which made the dashboard look healthy regardless of reality.
    """

    total_documents: int
    documents_by_status: Dict[str, int] = Field(default_factory=dict)
    indexed_chunks: int = 0
    total_words: int = 0
    avg_confidence: Optional[float] = Field(
        default=None, description="Mean AI confidence, or null when unmeasured"
    )
    feedback: Dict[str, Any] = Field(default_factory=dict)
    embeddings_are_real: bool = False
    ai_enrichment_available: bool = False


@router.get("/", tags=["Root"], summary="Service banner")
async def root() -> Dict[str, Any]:
    """Identify the service and point at its documentation."""
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "status": "operational",
        "docs": "/api/docs" if settings.docs_enabled else None,
        "health": "/api/health",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Component-level health",
)
async def health_check(
    db_manager: DatabaseManager = Depends(get_db_manager),
    retrieval: RetrievalService = Depends(get_retrieval_service),
    processor: DocumentProcessorService = Depends(get_document_processor),
) -> HealthCheckResponse:
    """Report the health of each dependency.

    Distinguishes *unhealthy* (cannot serve requests) from *degraded* (serving
    with reduced capability). A monitor that only sees "healthy" cannot tell
    that every search result is meaningless because no embedding model loaded.
    """
    settings = get_settings()
    components: Dict[str, ComponentHealth] = {}

    db_ok = await db_manager.health_check()
    components["database"] = ComponentHealth(
        status="ok" if db_ok else "unavailable",
        detail=None if db_ok else "Database connection failed",
    )

    if retrieval.embeddings_are_real:
        components["embeddings"] = ComponentHealth(status="ok")
    else:
        components["embeddings"] = ComponentHealth(
            status="degraded",
            detail=(
                "Keyword-only matching: "
                + (retrieval.embedding_service.degraded_reason or "no model loaded")
            ),
        )

    vector_stats = await retrieval.vector_store.get_stats()
    components["vector_store"] = ComponentHealth(
        status="ok" if vector_stats.get("initialized") else "unavailable",
        detail=f"backend={vector_stats.get('backend')}, "
        f"chunks={vector_stats.get('document_count')}",
    )

    if processor.ai_enabled:
        components["llm"] = ComponentHealth(status="ok")
    else:
        components["llm"] = ComponentHealth(
            status="degraded",
            detail=(
                f"No LLM reachable at {settings.ollama_base_url}. Documents are "
                "extracted and indexed, but not summarised."
            ),
        )

    statuses = {component.status for component in components.values()}
    if "unavailable" in statuses:
        overall = "unhealthy"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    return HealthCheckResponse(
        status=overall,
        version=settings.version,
        environment=settings.environment,
        components=components,
    )


@router.get("/health/live", tags=["System"], summary="Liveness probe")
async def liveness() -> Dict[str, str]:
    """Always 200 while the process is running.

    Kubernetes restarts a container that fails liveness, so this must not
    depend on the database — a transient DB outage should not trigger a
    restart loop.
    """
    return {"status": "alive"}


@router.get("/health/ready", tags=["System"], summary="Readiness probe")
async def readiness(
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> Dict[str, Any]:
    """Report whether this instance should receive traffic."""
    if not await db_manager.health_check():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        )
    return {"status": "ready"}


@router.get(
    "/metrics", tags=["System"], summary="Prometheus metrics", include_in_schema=False
)
async def prometheus_metrics() -> Response:
    """Expose the Prometheus registry.

    The metrics module and a Grafana dashboard both existed already, but
    nothing served the scrape endpoint, so no metric ever left the process.
    """
    settings = get_settings()
    if not settings.metrics_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Metrics are disabled"
        )

    from backend.observability.metrics import get_metrics

    metrics = get_metrics()
    return Response(content=metrics.get_metrics(), media_type=metrics.get_content_type())


@router.post("/feedback", tags=["Feedback"], summary="Rate an answer")
async def submit_feedback(
    request: FeedbackRequest,
    user: CurrentUser = Depends(get_current_user),
    feedback: FeedbackRepository = Depends(get_feedback_repository),
) -> Dict[str, Any]:
    """Record a thumbs up/down against an assistant message."""
    return await feedback.add(
        message_id=request.message_id,
        is_positive=request.is_positive,
        session_id=request.session_id,
        owner_id=user.user_id,
        comment=request.comment,
    )


@router.get(
    "/dashboard/metrics",
    response_model=DashboardMetricsResponse,
    tags=["Dashboard"],
    summary="Aggregate figures for the caller",
)
async def get_dashboard_metrics(
    user: CurrentUser = Depends(get_current_user),
    processor: DocumentProcessorService = Depends(get_document_processor),
    feedback: FeedbackRepository = Depends(get_feedback_repository),
    retrieval: RetrievalService = Depends(get_retrieval_service),
) -> DashboardMetricsResponse:
    """Return document and quality statistics for the signed-in user."""
    stats = await processor.stats(owner_id=user.user_id)
    feedback_stats = await feedback.stats(owner_id=user.user_id)
    avg_confidence = await _mean_confidence(processor, user.user_id)

    return DashboardMetricsResponse(
        total_documents=stats["total_documents"],
        documents_by_status=stats["by_status"],
        indexed_chunks=stats["total_chunks"],
        total_words=stats["total_words"],
        avg_confidence=avg_confidence,
        feedback=feedback_stats,
        embeddings_are_real=retrieval.embeddings_are_real,
        ai_enrichment_available=processor.ai_enabled,
    )


async def _mean_confidence(
    processor: DocumentProcessorService, owner_id: str
) -> Optional[float]:
    """Mean AI confidence across completed documents, or None if unmeasured."""
    documents, _ = await processor.list_documents(
        limit=500, status_filter="completed", owner_id=owner_id
    )
    scores = []
    for document in documents:
        confidence = (document.get("ai_insights") or {}).get("confidence")
        if isinstance(confidence, (int, float)):
            scores.append(float(confidence))

    return round(sum(scores) / len(scores), 4) if scores else None
