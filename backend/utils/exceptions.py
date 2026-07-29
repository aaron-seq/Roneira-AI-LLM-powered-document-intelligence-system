"""Application error types and the global exception handler.

The handler previously ``await``-ed structlog's synchronous ``logger.error()``.
Since that returns ``None``, every unhandled exception raised a second
``TypeError`` inside the handler — so callers got a bare ASGI 500 with no JSON
body and no correlation ID, precisely when diagnostics mattered most.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("backend.errors")


class AppError(Exception):
    """Base class for expected, operational application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.original_error = original_error


class ResourceNotFoundError(AppError):
    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            f"{resource} with id {resource_id} not found",
            status.HTTP_404_NOT_FOUND,
            details={"resource": resource, "resource_id": resource_id},
        )


class AuthenticationError(AppError):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)


class AuthorizationError(AppError):
    def __init__(self, message: str = "Not permitted"):
        super().__init__(message, status.HTTP_403_FORBIDDEN)


class ValidationError(AppError):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, details)


class ProcessingError(AppError):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            original_error=original_error,
        )


def _correlation_id(request: Request) -> str:
    """Read the correlation ID attached by the middleware."""
    return getattr(request.state, "correlation_id", "unknown")


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Return a structured response for a known application error."""
    correlation_id = _correlation_id(request)
    log = logger.warning if exc.status_code < 500 else logger.error

    log(
        "Application error: %s",
        exc.message,
        extra={
            "status_code": exc.status_code,
            "details": exc.details,
            "correlation_id": correlation_id,
            "path": request.url.path,
        },
        exc_info=exc.original_error or (exc if exc.status_code >= 500 else None),
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "correlation_id": correlation_id,
        },
        headers={"X-Correlation-ID": correlation_id},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a safe 500 for anything unexpected.

    The stack trace goes to the logs; the client gets a generic message plus a
    correlation ID it can quote. Leaking exception text to callers has handed
    out file paths, SQL and connection strings in more than one product.
    """
    correlation_id = _correlation_id(request)

    logger.error(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
        extra={"correlation_id": correlation_id, "path": request.url.path},
        exc_info=exc,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": (
                "An unexpected error occurred. Quote the correlation ID when "
                "reporting this."
            ),
            "details": {},
            "correlation_id": correlation_id,
        },
        headers={"X-Correlation-ID": correlation_id},
    )


# Backwards-compatible alias for the previous single-handler entry point.
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Dispatch to the appropriate handler based on the exception type."""
    if isinstance(exc, AppError):
        return await app_error_handler(request, exc)
    return await unhandled_exception_handler(request, exc)
