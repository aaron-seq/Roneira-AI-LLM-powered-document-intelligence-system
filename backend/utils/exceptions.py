from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger()

class AppError(Exception):
    """Base error class for the application."""

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
        )

class AuthenticationError(AppError):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)

class ValidationError(AppError):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, details)

class ProcessingError(AppError):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            original_error=original_error
        )

async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that ensures all errors are logged with stack traces
    and return structured JSON responses.
    """
    # Generate correlation ID (assuming middleware adds it, or generate one)
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    if isinstance(exc, AppError):
        # Operational errors (expected)
        # Log as warning if 4xx, error if 5xx
        log_method = logger.warning if exc.status_code < 500 else logger.error
        await log_method(
            "Application error",
            error=exc.message,
            status_code=exc.status_code,
            details=exc.details,
            correlation_id=correlation_id,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
                "correlation_id": correlation_id,
            },
        )
    
    # Unexpected errors
    import traceback
    stack_trace = traceback.format_exc()
    
    await logger.error(
        "Unhandled exception",
        error=str(exc),
        stack_trace=stack_trace,
        correlation_id=correlation_id,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred.",
            "correlation_id": correlation_id,
        },
    )
