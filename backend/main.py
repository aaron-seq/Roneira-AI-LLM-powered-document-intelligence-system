"""FastAPI application entry point.

Run with::

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from backend.api.dependencies import cleanup_services, initialize_services
from backend.api.middleware import (
    CorrelationIdMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from backend.api.routers import api_router
from backend.core.config import get_settings
from backend.utils.exceptions import (
    AppError,
    app_error_handler,
    unhandled_exception_handler,
)
from backend.utils.telemetry import setup_telemetry

settings = get_settings()
setup_telemetry(level=settings.log_level, json_logs=settings.is_production)
logger = logging.getLogger("backend.main")


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """Start services on boot and release them on shutdown."""
    logger.info(
        "Starting %s v%s (environment=%s)",
        settings.app_name,
        settings.version,
        settings.environment,
    )

    os.makedirs(settings.upload_directory, exist_ok=True)
    os.makedirs(settings.processed_files_directory, exist_ok=True)

    try:
        container = await initialize_services(app)
    except Exception:
        # Startup failures must be loud. A half-initialized app that answers
        # health checks is worse than one that refuses to start.
        logger.exception("Startup failed")
        raise

    if not container.retrieval_service.embeddings_are_real:
        logger.warning(
            "Search is running in KEYWORD-ONLY mode: no embedding model is "
            "loaded, so questions phrased differently from the source text "
            "will not match. Install sentence-transformers for semantic "
            "search, or set REQUIRE_REAL_EMBEDDINGS=true to make this fatal."
        )

    logger.info("Startup complete")
    try:
        yield
    finally:
        logger.info("Shutting down")
        await cleanup_services(app)


def create_application() -> FastAPI:
    """Build the FastAPI application.

    Middleware order matters: Starlette runs them outermost-first, so the
    correlation ID is established before anything else can log, and security
    headers are applied to every response including error responses.
    """
    application = FastAPI(
        title=settings.app_name,
        description=(
            "Upload documents, extract and index their contents, and ask "
            "questions answered with citations back to the source."
        ),
        version=settings.version,
        docs_url="/api/docs" if settings.docs_enabled else None,
        redoc_url="/api/redoc" if settings.docs_enabled else None,
        openapi_url="/api/openapi.json" if settings.docs_enabled else None,
        lifespan=application_lifespan,
    )

    application.add_middleware(CorrelationIdMiddleware)
    application.add_middleware(SecurityHeadersMiddleware, hsts=settings.is_production)

    if settings.metrics_enabled:
        application.add_middleware(MetricsMiddleware)

    # Host validation only helps when it is actually restrictive.
    if settings.allowed_hosts and "*" not in settings.allowed_hosts:
        application.add_middleware(
            TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts
        )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Correlation-ID",
            "X-Request-ID",
        ],
        expose_headers=["X-Correlation-ID", "X-Response-Time-ms"],
    )

    _install_rate_limiting(application)

    application.include_router(api_router, prefix="/api")

    application.add_exception_handler(AppError, app_error_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)

    return application


def _install_rate_limiting(application: FastAPI) -> None:
    """Attach per-client rate limiting.

    Uploads, chat and search get a tighter budget than plain reads because
    each one costs a document parse or an LLM inference.
    """
    application.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_per_minute,
        expensive_per_minute=settings.rate_limit_expensive_per_minute,
        expensive_paths=(
            "/api/documents/upload",
            "/api/chat",
            "/api/search",
        ),
        # Polled by infrastructure on a fixed schedule; throttling these would
        # make a busy service look down.
        exempt_paths=(
            "/api/health",
            "/api/health/live",
            "/api/health/ready",
            "/api/metrics",
        ),
    )
    logger.info(
        "Rate limiting enabled: %s req/min (%s req/min for upload, chat, search)",
        settings.rate_limit_per_minute,
        settings.rate_limit_expensive_per_minute,
    )


app = create_application()


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        # Binding all interfaces is required inside a container, where
        # localhost is not reachable from outside. Override with HOST when
        # running directly on a machine with other listeners.
        host=os.getenv("HOST", "0.0.0.0"),  # nosec B104
        port=int(os.getenv("PORT", "8000")),
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
